"""`llm-redact models`: the NER models a configuration loads.

``list`` and ``verify`` read the configuration (``--config`` like ``serve``
and ``doctor``) and the local files only — the Hugging Face cache, local
model folders — never the network: ``list`` shows each model of the
``gliner`` and ``hf`` backends with its revision, catalog facts and whether
its files are there; ``verify`` exits 1 unless every one is complete at its
revision (a cached snapshot can be incomplete), and ``verify --dir DIR``
checks a folder written by ``models pull --to`` against its manifest, for
use inside an air-gapped enclave (AD11). spaCy, Presidio and Stanza models
are not Hugging Face snapshots: their install commands are printed instead.

Output names backends, model ids, revisions, files and catalog facts only.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from llm_redact.config import Config, ConfigError, apply_env_overrides, load_config

# Exit codes: 0 fine, 1 a model (or folder) is not complete, 2 the command
# could not run (an unreadable configuration, bad arguments).
OK, FAILED, UNUSABLE = 0, 1, 2


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    models = subparsers.add_parser(
        "models", help="list, verify or pull the NER models the config loads (gliner, hf)"
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


def run_models(args: argparse.Namespace) -> int:
    if args.models_command == "verify":
        if args.dir is not None:
            return run_verify_dir(args)
        return run_verify(args)
    return run_list(args)


def _config(args: argparse.Namespace) -> Config | None:
    try:
        return apply_env_overrides(load_config(args.config))
    except ConfigError as problem:
        print(f"llm-redact models: {problem}", file=sys.stderr)
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
        state["problem"] = "huggingface_hub is not installed (the hf and gliner extras install it)"
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
        print("no Hugging Face Hub models configured (backends gliner, hf)")
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
        print("no Hugging Face Hub models configured (backends gliner, hf): nothing to verify")
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
    from llm_redact.detection.model_files import gliner_files, hf_files

    folder = str(root / model.folder)
    try:
        if model.backend == "hf":
            hf_files(folder, revision=None, allow_download=False, allow_pickle_weights=True)
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
