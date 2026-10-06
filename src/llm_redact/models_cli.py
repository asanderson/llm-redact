"""`llm-redact models`: the NER models a configuration loads.

``list`` and ``verify`` read the configuration (``--config`` like ``serve``
and ``doctor``) and the local files only — the Hugging Face cache, local
model folders — never the network: ``list`` shows each model of the
``gliner``, ``gliner2`` and ``hf`` backends with its revision, catalog facts and whether
its files are there; ``verify`` exits 1 unless every one is complete at its
revision (a cached snapshot can be incomplete), and ``verify --dir DIR``
checks a folder written by ``models pull --to`` against its manifest, for
use inside an air-gapped enclave (AD11). ``pull`` is the one subcommand
that downloads: each model (and GLiNER base model) at its revision, with
the loaders' own file names, into the Hugging Face cache — and with ``--to``
also into portable, self-contained folders with a SHA-256 manifest. spaCy,
Presidio and Stanza models are not Hugging Face snapshots: their install
commands are printed instead.

Output names backends, model ids, revisions, files and catalog facts only.
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from llm_redact.config import (
    Config,
    ConfigError,
    apply_env_overrides,
    load_config,
    resolve_config_path,
)

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig
    from llm_redact.detection.model_files import ModelFiles
    from llm_redact.detection.model_manifest import ManifestModel
    from llm_redact.detection.model_sources import ModelSource

# Exit codes: 0 fine, 1 a model (or folder) is not complete, 2 the command
# could not run (an unreadable configuration, bad arguments).
OK, FAILED, UNUSABLE = 0, 1, 2


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    models = subparsers.add_parser(
        "models", help="list, verify or pull the NER models the config loads (gliner, gliner2, hf)"
    )
    sub = models.add_subparsers(dest="models_command", required=True)
    listing = sub.add_parser(
        "list",
        help="each configured model: revision, catalog status, license, cached (offline)",
    )
    listing.add_argument("--config", type=Path, default=None, help="path to config.toml")
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    verify = sub.add_parser(
        "verify",
        help="exit 1 unless every configured model's files are complete at its revision"
        " (offline); --dir checks a folder written by `models pull --to`",
    )
    verify.add_argument("--config", type=Path, default=None, help="path to config.toml")
    verify.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="check DIR against its llm-redact-models.json (sizes and SHA-256; no config)",
    )
    verify.add_argument("--json", action="store_true", help="machine-readable output")
    pull = sub.add_parser(
        "pull",
        help="download each configured model (and GLiNER base model) at its revision into the"
        " Hugging Face cache",
    )
    pull.add_argument("--config", type=Path, default=None, help="path to config.toml")
    pull.add_argument(
        "--to",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write portable, self-contained model folders and a SHA-256 manifest to DIR",
    )
    pull.add_argument(
        "--as",
        dest="mount_as",
        default=None,
        metavar="PATH",
        help="with --to: print the config snippet for DIR mounted at the absolute PATH (for"
        " example /models)",
    )


def run_models(args: argparse.Namespace) -> int:
    if args.models_command == "pull":
        return run_pull(args)
    if args.models_command == "verify":
        if args.dir is not None:
            return run_verify_dir(args)
        return run_verify(args)
    return run_list(args)


def _config(args: argparse.Namespace) -> Config | None:
    """The configuration ``--config`` names (else the one ``serve`` would
    find), or None — after saying why on stderr — when it cannot be read
    (exit 2): a missing or unreadable file as much as an invalid one."""
    try:
        # One resolution: the file a message names is the one read.
        source = args.config if args.config is not None else resolve_config_path()
    except ConfigError as problem:  # LLM_REDACT_CONFIG names no file
        print(f"llm-redact models: {problem}", file=sys.stderr)
        return None
    try:
        return apply_env_overrides(load_config(source))
    except ConfigError as problem:
        print(f"llm-redact models: {problem}", file=sys.stderr)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"llm-redact models: cannot read {source} ({type(exc).__name__})", file=sys.stderr)
    return None


def _json(value: object) -> None:
    from llm_redact.jsonwalk import json_text

    print(json_text(value))


def _hub_installed() -> bool:
    try:
        import huggingface_hub  # noqa: F401  (local cache lookups only)
    except ImportError:
        return False
    return True


# spaCy, Presidio and Stanza models are packages or library downloads,
# never Hugging Face snapshots: how to install each backend's model.
def other_models(config: Config) -> list[dict[str, str]]:
    """The models of the active backends that are not Hub snapshots, each
    with the command that installs it."""
    from llm_redact.detection.model_catalog import HUB_BACKENDS

    ner = config.detection.ner
    found = []
    for backend in ner.active_backends():
        if backend in HUB_BACKENDS:
            continue
        if backend == "stanza":
            language = ner.language or "en"
            model = f"stanza {language}"
            install = f"uv run python -c \"import stanza; stanza.download('{language}')\""
        else:  # spacy, presidio: a spaCy pipeline package
            model = ner.model_for(backend) or "en_core_web_sm"
            install = f"uv run python -m spacy download {model}"
        found.append({"backend": backend, "model": model, "install": install})
    return found


def _print_others(config: Config) -> None:
    for other in other_models(config):
        print(
            f"{other['backend']}: {other['model']} is not a Hugging Face model; install it"
            f" with: {other['install']}"
        )


def model_state(config: Config, source: Any, *, hub: bool) -> dict[str, Any]:
    """One Hub backend's model as `models list` and `verify` report it:
    its source facts and whether its files are complete where its load
    reads them (``files``: cached / folder / missing / incomplete / error /
    unchecked), from local files only."""
    from llm_redact.detection.model_files import ModelNotCached
    from llm_redact.detection.model_sources import local_files

    entry = source.entry
    state: dict[str, Any] = {
        "backend": source.backend,
        "model": source.model,
        "source": "local" if source.local else "hub",
        "model_id": source.model_id,
        "revision": source.revision,
        "pinned_by": source.pinned_by,
        "catalog": entry.status if entry is not None else None,
        "license": entry.license if entry is not None else None,
        "facts": entry.describe() if entry is not None else None,
        "min_versions": [list(pair) for pair in entry.min_versions] if entry is not None else [],
        # Until the files say otherwise: the base model the catalog pins (it
        # pins one only where the checkpoint ships no tokenizer of its own).
        "backbone": entry.backbone if entry is not None and entry.backbone_revision else None,
        "backbone_revision": entry.backbone_revision if entry is not None else None,
        "files": "unchecked",
        "file_count": None,
        "problem": source.sidecar_problem,
    }
    if source.sidecar_problem is not None:
        state["files"] = "error"
        return state
    if not hub and not source.local:
        state["problem"] = (
            "huggingface_hub is not installed (the hf, gliner and gliner2 extras install it)"
        )
        return state
    try:
        files = local_files(config.detection.ner, source)
    except ModelNotCached as missing:
        state["files"] = "incomplete" if missing.missing else "missing"
        state["problem"] = str(missing)
        return state
    except ConfigError as problem:
        state["files"] = "error"
        state["problem"] = str(problem)
        return state
    state["files"] = "folder" if source.local else "cached"
    state["file_count"] = len(files.files) + (files.config is not None)
    state["backbone"] = files.backbone
    state["backbone_revision"] = files.backbone_revision
    return state


def _short(revision: object) -> str:
    return str(revision)[:12] if isinstance(revision, str) else "-"


def _table(rows: list[list[str]]) -> None:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


def _states(config: Config) -> list[dict[str, Any]]:
    from llm_redact.detection.model_sources import hub_sources

    hub = _hub_installed()
    return [model_state(config, source, hub=hub) for source in hub_sources(config.detection.ner)]


def run_list(args: argparse.Namespace) -> int:
    config = _config(args)
    if config is None:
        return UNUSABLE
    states = _states(config)
    if args.json:
        _json(
            {
                "ner_enabled": config.detection.ner.enabled,
                "models": states,
                "other_models": other_models(config),
            }
        )
        return OK
    if not config.detection.ner.enabled:
        print("NER is off ([detection.ner] enabled = false); the models it would load:")
    if states:
        rows = [["BACKEND", "MODEL", "REVISION", "CATALOG", "LICENSE", "FILES", "BASE MODEL"]]
        for state in states:
            backbone = state["backbone"]
            rows.append(
                [
                    state["backend"],
                    state["model"],
                    _short(state["revision"]),
                    state["catalog"] or "-",
                    state["license"] or "-",
                    state["files"],
                    f"{backbone}@{_short(state['backbone_revision'])}" if backbone else "-",
                ]
            )
        _table(rows)
        for state in states:
            if state["problem"] is not None:
                print(f"{state['backend']}: {state['problem']}")
    else:
        print("no Hugging Face Hub models configured (backends gliner, gliner2, hf)")
    _print_others(config)
    return OK


def run_verify(args: argparse.Namespace) -> int:
    config = _config(args)
    if config is None:
        return UNUSABLE
    states = _states(config)
    good = all(state["files"] in ("cached", "folder") for state in states)
    if args.json:
        _json({"ok": good, "models": states, "other_models": other_models(config)})
        return OK if good else FAILED
    if not states:
        print(
            "no Hugging Face Hub models configured (backends gliner, gliner2, hf):"
            " nothing to verify"
        )
    for state in states:
        where = f"{state['backend']}: {state['model']}"
        if state["files"] in ("cached", "folder"):
            place = "its folder" if state["files"] == "folder" else "the local Hugging Face cache"
            at = f" at {state['revision']}" if state["revision"] else ""
            print(f"OK    {where}{at}: {state['file_count']} files complete in {place}")
        else:
            print(f"FAIL  {where}: {state['problem']}")
    _print_others(config)
    return OK if good else FAILED


def run_verify_dir(args: argparse.Namespace) -> int:
    """Check a folder written by `models pull --to` against its manifest:
    every file's size and SHA-256, no unlisted file in a model folder, each
    sidecar naming its model, and every folder complete for its loader.
    No configuration and no network."""
    from llm_redact.detection.model_manifest import (
        MANIFEST_NAME,
        ManifestError,
        check_files,
        read_manifest,
    )

    root = args.dir
    try:
        models = read_manifest(root)
    except ManifestError as problem:
        if args.json:
            _json({"ok": False, "problem": str(problem), "models": []})
        else:
            print(f"FAIL  {problem}")
        return FAILED
    results = []
    for model in models:
        check = check_files(root, model)
        if not check.problems:
            check.problems += _loader_problems(root, model)
        results.append(check)
    good = all(not check.problems for check in results)
    folders = {model.folder for model in models}
    extra = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.name not in folders and entry.name != MANIFEST_NAME
    )
    if args.json:
        _json(
            {
                "ok": good,
                "models": [
                    {
                        "backend": check.model.backend,
                        "model_id": check.model.model_id,
                        "revision": check.model.revision,
                        "folder": check.model.folder,
                        "problems": check.problems,
                    }
                    for check in results
                ],
                "unlisted_entries": extra,
            }
        )
        return OK if good else FAILED
    for check in results:
        model = check.model
        where = f"{model.backend}: {model.model_id} at {model.revision or 'an unrecorded revision'}"
        if check.problems:
            for found in check.problems:
                print(f"FAIL  {where}: {found}")
        else:
            print(
                f"OK    {where}: {model.folder} ({len(model.files)} files) matches {MANIFEST_NAME}"
            )
    if extra:
        print(f"note: {root} also holds entries the manifest does not list: {', '.join(extra)}")
    if not models:
        print(f"note: {MANIFEST_NAME} lists no model")
    return OK if good else FAILED


def _loader_problems(root: Path, model: Any) -> list[str]:
    """Whether the folder holds what its loader reads (a manifest could
    list an incomplete folder): the loaders' own check, files only."""
    from llm_redact.detection.model_catalog import lookup
    from llm_redact.detection.model_files import gliner2_files, gliner_files, hf_files

    folder = str(root / model.folder)
    entry = lookup(model.model_id)
    try:
        if model.backend == "hf":
            hf_files(
                folder,
                revision=None,
                allow_download=False,
                allow_pickle_weights=True,
                extra_files=entry.extra_files("hf") if entry is not None else (),
            )
        elif model.backend == "gliner2":
            gliner2_files(folder, revision=None, allow_download=False, check_types=False)
        else:
            gliner_files(
                folder,
                revision=None,
                allow_download=False,
                onnx_file=model.onnx,
                check_types=False,
            )
    except ConfigError as problem:
        return [str(problem)]
    return []


# --- pull ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pulled:
    """One model `models pull` fetched: its source, the commit it fetched
    and the files its load reads."""

    source: "ModelSource"
    revision: str | None
    files: "ModelFiles"
    # [detection.ner.onnx]: the ONNX file the folder carries instead of
    # torch weights (gliner only).
    onnx: str | None = None


class PullError(Exception):
    """A pull that cannot write its output (names paths only)."""


def snapshot_commit(path: Path | None) -> str | None:
    """The commit a Hugging Face cache snapshot holds: the cache keeps each
    revision in ``snapshots/<commit>``. None when the folder is not named
    like one."""
    from llm_redact.detection.model_catalog import REVISION_RE

    return path.name if path is not None and REVISION_RE.fullmatch(path.name) else None


def folder_name(backend: str, model_id: str) -> str:
    """A portable model folder's name: the backend and the model id with
    every run of other characters as ``--`` (``hf-dslim--bert-base-NER``)."""
    return f"{backend}-{re.sub(r'[^A-Za-z0-9._-]+', '--', model_id)}"


def absolute_mount(path: str) -> bool:
    """Whether ``--as`` names an absolute path (POSIX or Windows: the
    machine that mounts the folders need not be this one). A relative one
    is read against the proxy's working directory, and the loader reads a
    value like ``models/hf-dslim--bert-base-NER`` as a Hugging Face model
    id wherever that directory has no such folder."""
    return PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()


def run_pull(args: argparse.Namespace) -> int:
    if args.mount_as is not None and args.to is None:
        print("llm-redact models pull: --as needs --to", file=sys.stderr)
        return UNUSABLE
    if args.mount_as is not None and not absolute_mount(args.mount_as):
        print(
            "llm-redact models pull: --as needs an absolute path: where DIR is mounted for the"
            " proxy (for example /models)",
            file=sys.stderr,
        )
        return UNUSABLE
    config = _config(args)
    if config is None:
        return UNUSABLE
    from llm_redact.detection.model_sources import hub_sources

    ner = config.detection.ner
    sources = hub_sources(ner)
    if not ner.enabled:
        print("NER is off ([detection.ner] enabled = false); pulling the models it would load")
    _print_others(config)
    if not sources:
        print(
            "no Hugging Face Hub models configured (backends gliner, gliner2, hf): nothing to pull"
        )
        return OK
    if not _hub_installed():
        print(
            "FAIL  pulling needs huggingface_hub, which the hf, gliner and gliner2 extras install;"
            " install the backend's extra (uv sync --extra hf)"
        )
        return FAILED
    pulled: list[Pulled] = []
    failed = False
    for source in sources:
        ok, item = _pull_one(ner, source)
        failed = failed or not ok
        if item is not None:
            pulled.append(item)
    if failed:
        if args.to is not None:
            print(f"FAIL  not every model was pulled; nothing was written to {args.to}")
        return FAILED
    if args.to is None:
        return OK
    if not pulled:
        print(
            f"note: nothing was written to {args.to}: `--to` copies the Hub models pull fetches,"
            " never a local folder"
        )
        return OK
    try:
        written = write_portable(args.to, pulled)
    except PullError as problem:
        print(f"FAIL  {problem}")
        return FAILED
    except OSError as exc:
        print(f"FAIL  cannot write the model folders to {args.to} ({type(exc).__name__})")
        return FAILED
    _print_snippet(ner, args, written)
    return OK


def _pull_one(ner: "NerConfig", source: "ModelSource") -> tuple[bool, Pulled | None]:
    """Fetch one model (and its base model) at its revision; print what
    happened. (ok, the pulled model or None)."""
    from llm_redact.detection.model_sources import local_files

    where = f"{source.backend}: {source.model}"
    if source.sidecar_problem is not None:
        print(f"FAIL  {source.backend}: {source.sidecar_problem}")
        return False, None
    if source.local:
        return _pull_local(ner, source, where), None
    restricted = source.restricted_warning()
    if restricted is not None:
        print(f"warning: {restricted}")
    revision = ner.revision_for(source.backend)
    try:
        files = local_files(ner, source, revision=revision, allow_download=True)
    except ConfigError as problem:
        print(f"FAIL  {problem}")
        return False, None
    if revision is None:
        revision = snapshot_commit(files.directory)
        if revision is not None:
            print(
                f"note: {where} has no pin; its default branch is at {revision}. Pin it:"
                f' [detection.ner.revisions] {source.backend} = "{revision}"'
            )
        else:
            print(
                f"note: {where} has no pin, and the commit pulled could not be told; pin one"
                f" in [detection.ner.revisions] {source.backend}"
            )
    files = _unpinned_base(where, files)
    count = len(files.files) + (files.config is not None)
    print(f"OK    {where} at {revision or _DEFAULT_BRANCH}: {count} files{_base_text(files)}")
    return True, Pulled(source, revision, files, ner.onnx_for(source.backend))


def _pull_local(ner: "NerConfig", source: "ModelSource", where: str) -> bool:
    """A local folder is read, never fetched (nor copied by ``--to``) —
    but a GLiNER folder that ships no tokenizer or ``encoder_config`` (a
    clone of an urchade model) loads with its base model's from the
    Hugging Face cache, and the startup and ``verify`` name ``models pull``
    while that base model is missing: fetch it (and only it). A folder
    that cannot load fails as the startup would. Returns ok."""
    from llm_redact.detection.model_sources import local_files

    if source.backend == "gliner":
        try:
            files = local_files(ner, source, allow_download=True)
        except ConfigError as problem:
            print(f"FAIL  {problem}")
            return False
        if files.backbone is not None and not _local_base(files):
            files = _unpinned_base(where, files)
            print(
                f"OK    {where} (a local folder): base model {files.backbone} at"
                f" {files.backbone_revision or _DEFAULT_BRANCH}"
            )
            return True
    print(f"skip  {where} is a local folder; nothing to pull")
    return True


# What a revision `models pull` fetched without a pin, and could not name, is.
_DEFAULT_BRANCH = "its default branch"


def _local_base(files: "ModelFiles") -> bool:
    """Whether ``files``' base model is a local folder: read as it is,
    nothing to pull and no pin — never a Hub snapshot, whatever its name."""
    from llm_redact.detection.model_files import is_local

    return files.backbone is not None and is_local(files.backbone)


def _unpinned_base(where: str, files: "ModelFiles") -> "ModelFiles":
    """``files`` with the commit of a Hub base model the catalog does not
    pin, read from the cache folder it was pulled into, and a note saying
    so."""
    if files.backbone is None or files.backbone_revision is not None or _local_base(files):
        return files
    commit = snapshot_commit(files.backbone_directory)
    print(
        f"note: {where}: the model catalog pins no revision of its base model"
        f" {files.backbone}; pulled {commit or _DEFAULT_BRANCH}"
    )
    return replace(files, backbone_revision=commit)


def _base_text(files: "ModelFiles") -> str:
    """The base model, as the end of a pulled model's OK line."""
    if files.backbone is None:
        return ""
    if _local_base(files):
        return f"; base model {files.backbone} (a local folder)"
    return f"; base model {files.backbone} at {files.backbone_revision or _DEFAULT_BRANCH}"


def write_portable(root: Path, pulled: list[Pulled]) -> list[tuple[str, str]]:
    """Write each pulled model as a self-contained folder under ``root``
    (its files, an assembled GLiNER model's gliner_config.json, and its
    ``llm-redact-model.json``), then the manifest. Returns (backend, folder)
    pairs. A folder of the same name is replaced only when it is one this
    command wrote (it holds a sidecar).

    Nothing is replaced until every folder it would replace is checked and
    every new folder (and the new manifest) is written beside the old ones:
    a refusal, or a failure while writing, leaves ``root`` as it was. The
    old manifest goes before the first folder is replaced and the new one
    comes after the last, so ``root`` never holds a manifest describing
    folders it no longer holds (a failure while replacing leaves none, and
    ``models verify --dir`` fails until a pull completes)."""
    from llm_redact import __version__
    from llm_redact.detection.model_manifest import MANIFEST_NAME, manifest_json
    from llm_redact.jsonwalk import json_text

    root.mkdir(parents=True, exist_ok=True)
    named = []
    for item in pulled:
        model_id = item.source.model_id or item.source.model
        name = folder_name(item.source.backend, model_id)
        _require_replaceable(root / name)
        named.append((name, model_id, item))
    staged: list[Path] = []
    manifest = root / f".{MANIFEST_NAME}.partial"
    try:
        models = []
        for name, model_id, item in named:
            temp = root / f".{name}.partial"
            staged.append(temp)
            models.append(_stage_folder(temp, name, model_id, item))
        manifest.write_text(json_text(manifest_json(models, __version__)) + "\n", encoding="utf-8")
        (root / MANIFEST_NAME).unlink(missing_ok=True)
        for temp, (name, _, _) in zip(staged, named, strict=True):
            _replace_folder(temp, root / name)
        os.replace(manifest, root / MANIFEST_NAME)
    finally:
        for temp in staged:
            shutil.rmtree(temp, ignore_errors=True)
        manifest.unlink(missing_ok=True)
    return [(item.source.backend, name) for name, _, item in named]


def _require_replaceable(final: Path) -> None:
    """Refuse to replace anything but a folder `pull --to` wrote (one that
    holds a sidecar)."""
    from llm_redact.detection.model_catalog import SIDECAR_NAME

    if final.exists() and not (final / SIDECAR_NAME).is_file():
        raise PullError(
            f"{final} exists and is not a folder `llm-redact models pull --to` wrote; remove it"
            " or choose another --to"
        )


def _replace_folder(temp: Path, final: Path) -> None:
    """Put the folder written at ``temp`` in the place of ``final``."""
    if final.exists():
        shutil.rmtree(final)
    os.rename(temp, final)


def _stage_folder(temp: Path, name: str, model_id: str, item: Pulled) -> "ManifestModel":
    """Write ``item``'s folder at ``temp`` (copies, never links) and return
    its manifest entry, as the folder will be named."""
    from llm_redact.detection.model_catalog import SIDECAR_NAME, ModelIdentity, sidecar_text
    from llm_redact.detection.model_files import GLINER_CONFIG, config_text
    from llm_redact.detection.model_manifest import ManifestModel, file_records, folder_files

    shutil.rmtree(temp, ignore_errors=True)  # (left by a run that was killed)
    temp.mkdir()
    for relative, source in item.files.files.items():
        target = temp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.realpath(source), target)
    if item.files.config is not None:
        (temp / GLINER_CONFIG).write_text(config_text(item.files.config), encoding="utf-8")
    identity = ModelIdentity(model_id, item.revision)
    (temp / SIDECAR_NAME).write_text(sidecar_text(identity), encoding="utf-8")
    records = file_records(temp, folder_files(temp))
    # The manifest names a Hub base model only: a local one is no Hub id,
    # whatever its name (its files are in the folder, each with its SHA-256).
    hub_base = not _local_base(item.files)
    return ManifestModel(
        backend=item.source.backend,
        model_id=model_id,
        revision=item.revision,
        folder=name,
        files=tuple((r["path"], r["size"], r["sha256"]) for r in records),
        backbone=item.files.backbone if hub_base else None,
        backbone_revision=item.files.backbone_revision if hub_base else None,
        onnx=item.onnx,
    )


def _print_snippet(
    ner: "NerConfig", args: argparse.Namespace, written: list[tuple[str, str]]
) -> None:
    """The [detection.ner.models] entries that load the written folders,
    as they will be mounted (``--as``) or where they are."""
    from llm_redact.config_write import _toml_str
    from llm_redact.detection.model_manifest import MANIFEST_NAME

    def place(name: str) -> str:
        if args.mount_as is not None:
            return f"{args.mount_as.rstrip('/')}/{name}"
        return str((args.to / name).resolve())

    where = args.mount_as if args.mount_as is not None else str(args.to.resolve())
    print(
        f"wrote {len(written)} model folder(s) and {MANIFEST_NAME} to {args.to}; to load them"
        f" from {where}, set in the configuration:"
    )
    print()
    print("[detection.ner.models]")
    for backend, name in sorted(written):
        print(f"{backend} = {_toml_str(place(name))}")
    print()
    pinned = sorted(backend for backend, _ in written if ner.configured_revision(backend))
    if pinned:
        print(
            "and remove these backends' [detection.ner.revisions] entries (a local folder takes"
            f" none: its llm-redact-model.json records the revision): {', '.join(pinned)}"
        )
    print(f"check the folder where it is used with: llm-redact models verify --dir {where}")
