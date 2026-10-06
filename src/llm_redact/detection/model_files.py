"""Where NER model files come from: local, pinned, and free of model code.

The ``hf``, ``gliner`` and ``gliner2`` backends load Hugging Face Hub models. This module
turns a configured model value into a local directory the loaders read, and
nothing else ever fetches a file:

* a local directory is used as it is (a revision then applies to nothing,
  so configuring one is an error);
* a Hub model id is resolved with ``huggingface_hub.snapshot_download`` at
  its pinned revision (``[detection.ner.revisions]``, else the model
  catalog's pin), with an explicit list of top-level file names
  (``allow_patterns``) — never TensorFlow, Flax, ONNX or ``original/``
  copies a repository may also hold — and ``local_files_only`` unless
  ``[detection.ner] allow_download`` is set AND the build is the process's
  startup (``serve``, ``serve --check``): a reload, a config dry run or a
  preview never downloads, and a build that may not download first sets
  the libraries' offline switches (:func:`go_offline`);
* a model whose configuration names code to run (``auto_map``) is refused:
  llm-redact never loads with ``trust_remote_code``;
* ``hf`` weights must be safetensors unless ``allow_pickle_weights`` is set
  (owner decision D3): a ``pytorch_model.bin`` is a pickle, and a pickle can
  run code when it is loaded;
* a GLiNER checkpoint that ships no tokenizer or ``encoder_config`` (the
  urchade v2.1 models) is assembled into a self-contained folder with its
  pinned base model's tokenizer and configuration (owner decision D13), so
  GLiNER never fetches the base model from the Hub at load time; a GLiNER2
  checkpoint is self-contained by design and must be.

Messages name the backend, the model id and the revision only.
"""

import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_redact.config import ConfigError
from llm_redact.detection.model_catalog import MODEL_ID_RE, ModelIdentity

# Tokenizer files, by the names transformers looks for: a fast tokenizer's
# tokenizer.json, its configuration, and the vocabularies a slow tokenizer
# is converted from (WordPiece vocab.txt, byte-level BPE vocab.json +
# merges.txt, SentencePiece models).
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
)
# The files a tokenizer is built from: any one of these groups (its
# tokenizer_config.json and special-tokens maps only configure one).
VOCABULARIES: tuple[tuple[str, ...], ...] = (
    ("tokenizer.json",),
    ("vocab.txt",),
    ("vocab.json", "merges.txt"),
    ("spm.model",),
    ("sentencepiece.bpe.model",),
    ("spiece.model",),
    ("tokenizer.model",),
)
# safetensors weights: one file, or shards with their index.
SAFETENSORS_FILES = ("model.safetensors", "model.safetensors.index.json")
SAFETENSORS_SHARDS = "model-*.safetensors"
# Pickle weights, fetched only when a model has no safetensors and
# allow_pickle_weights is set.
PICKLE_FILES = ("pytorch_model.bin", "pytorch_model.bin.index.json")
PICKLE_SHARDS = "pytorch_model-*.bin"

# What an `hf` token-classification model needs: its configuration, its
# weights and its tokenizer. (snapshot_download's patterns are fnmatch
# patterns whose `*` also matches "/": every name here is top level.)
HF_PATTERNS = ("config.json", *SAFETENSORS_FILES, SAFETENSORS_SHARDS, *TOKENIZER_FILES)
HF_PICKLE_PATTERNS = (*PICKLE_FILES, PICKLE_SHARDS)

# A configuration file is read whole; none is near this size.
MAX_CONFIG_BYTES = 4 * 1024 * 1024


def _config_error(message: str) -> Exception:
    return ConfigError(message)


class ModelNotCached(ConfigError):
    """A Hub model (or a GLiNER model's base model) that the local Hugging
    Face cache does not hold, or holds only in part, at the revision a
    build without downloads asks for. ``missing`` names the kinds of files
    the cached snapshot lacks (empty: no snapshot at all)."""

    def __init__(
        self, what: str, model: str, revision: str | None, missing: tuple[str, ...] = ()
    ) -> None:
        text = (
            f"[detection.ner] {what} {model!r} {_revision_text(revision)} is not"
            " (completely) in the local Hugging Face cache, and downloads are off"
            " (an older revision in the cache does not count); run"
            " `llm-redact models pull`, or set [detection.ner] allow_download = true"
            " to fetch it at startup (a reload never downloads)"
        )
        super().__init__(f"{text}; missing: {', '.join(missing)}" if missing else text)
        self.what = what
        self.model = model
        self.revision = revision
        self.missing = missing


def is_local(model: str) -> bool:
    """Whether a configured model value names a local directory."""
    return Path(model).is_dir()


# The Hugging Face libraries' offline switches. huggingface_hub reads
# HF_HUB_OFFLINE when it is imported (into huggingface_hub.constants),
# transformers asks that constant; both refuse every request while set.
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def go_offline() -> None:
    """Make the Hugging Face libraries refuse every network request in this
    process from now on: the loaders' own lookups already pass
    ``local_files_only``, and this guards the library code they do not
    control. Set before the first NER import of a build that may not
    download (every reload; a startup with ``allow_download = false``) and
    never unset — no later build of this process downloads either."""
    for name in OFFLINE_VARIABLES:
        os.environ[name] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:  # imported before: the variable is read already
        setattr(constants, "HF_HUB_OFFLINE", True)  # noqa: B010 (a module, typed loosely)


def _revision_text(revision: str | None) -> str:
    return f"at revision {revision}" if revision is not None else "(no revision pinned)"


def resolve_model(
    model: str,
    *,
    what: str,
    revision: str | None,
    allow_download: bool,
    allow_patterns: Sequence[str],
) -> Path:
    """The local directory holding ``model``'s files.

    ``what`` names the model in messages ("hf model", "gliner model", ...).
    A local directory is returned as it is; any ``revision`` for one is an
    error (``NerConfig.revision_for`` never hands a folder a catalog pin,
    so a revision here is one the configuration names). A Hub model id is
    looked up at ``revision`` with ``allow_patterns``, from the local
    Hugging Face cache only unless ``allow_download``.
    """
    if is_local(model):
        if revision is not None:
            raise _config_error(
                f"[detection.ner] {what} {model!r} is a local directory; a revision in"
                " [detection.ner.revisions] applies only to a Hugging Face model id"
            )
        return Path(model)
    if not MODEL_ID_RE.fullmatch(model):
        raise _config_error(
            f"[detection.ner] {what} {model!r} is neither a local directory nor a"
            " Hugging Face model id"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise _config_error(
            f"[detection.ner] {what} {model!r} needs huggingface_hub, which the hf, gliner"
            " and gliner2 extras install; install the backend's extra"
        ) from exc
    try:
        path: str = snapshot_download(
            model,
            revision=revision,
            allow_patterns=list(allow_patterns),
            local_files_only=not allow_download,
        )
    except Exception as exc:  # not cached, incomplete, not found, no network
        if not allow_download:
            raise ModelNotCached(what, model, revision) from exc
        raise _config_error(
            f"[detection.ner] {what} {model!r} {_revision_text(revision)} could not be"
            f" fetched from the Hugging Face Hub: {type(exc).__name__}"
        ) from exc
    return Path(path)


def read_config(path: Path, *, what: str, model: str) -> dict[str, Any]:
    """A model directory's JSON configuration file as an object; a file
    that is too large or not a JSON object is a configuration error."""
    from llm_redact.jsonwalk import loads_bounded

    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {path.name} cannot be read ({type(exc).__name__})"
        ) from exc
    try:
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("too large")
        data = loads_bounded(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError, JSONDecodeError, JsonTooDeep
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {path.name} is not a JSON configuration"
        ) from exc
    if not isinstance(data, dict):
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {path.name} is not a JSON configuration"
        )
    return data


def _carries_code(value: object) -> bool:
    """Whether a configuration names code to load (an ``auto_map`` at any
    depth: transformers would import classes from the model repository)."""
    if isinstance(value, Mapping):
        return "auto_map" in value or any(_carries_code(item) for item in value.values())
    if isinstance(value, list):
        return any(_carries_code(item) for item in value)
    return False


def refuse_model_code(config: Mapping[str, Any], *, what: str, model: str, name: str) -> None:
    """Refuse a configuration that names code from the model repository."""
    if _carries_code(config):
        raise _config_error(
            f"[detection.ner] {what} {model!r} needs code from its repository ({name}"
            " names auto_map); llm-redact never runs model code"
        )


# The JSON files a load reads that can name code (a model's or a
# tokenizer's auto_map).
CODE_CONFIG_FILES = ("config.json", "tokenizer_config.json")


def check_configs(
    directory: Path, names: Iterable[str], *, what: str, model: str
) -> dict[str, dict[str, Any]]:
    """Each of the JSON configuration files ``names`` in ``directory``,
    read and checked for model code (``refuse_model_code``), by name; a
    file that is absent is left out."""
    configs: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        if path.is_file():
            config = read_config(path, what=what, model=model)
            refuse_model_code(config, what=what, model=model, name=name)
            configs[name] = config
    return configs


def has_files(directory: Path, names: Iterable[str]) -> bool:
    """Whether ``directory`` holds any of the files ``names``."""
    return any((directory / name).is_file() for name in names)


def has_vocabulary(names: Collection[str]) -> bool:
    """Whether the file ``names`` include a tokenizer's vocabulary (one of
    :data:`VOCABULARIES`)."""
    return any(all(name in names for name in group) for group in VOCABULARIES)


def matching_files(directory: Path, patterns: Iterable[str]) -> dict[str, Path]:
    """The top-level files of ``directory`` whose names match one of the
    file ``patterns`` (the names a lookup with those patterns fetches), by
    name, sorted."""
    patterns = tuple(patterns)
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return {}
    return {
        entry.name: entry
        for entry in entries
        if entry.is_file() and any(fnmatch.fnmatchcase(entry.name, p) for p in patterns)
    }


@dataclass(frozen=True)
class ModelFiles:
    """Every file one configured model's load reads, by its path inside a
    self-contained folder: what ``llm-redact models verify`` checks and
    ``models pull --to`` copies. Names only, never content."""

    model: str
    # Where the model's own files are read from: its snapshot in the
    # Hugging Face cache, or a local folder.
    directory: Path
    # Relative POSIX path inside the folder -> the file it is read from.
    files: Mapping[str, Path]
    # A gliner_config.json llm-redact writes for an ASSEMBLED folder (the
    # checkpoint's own configuration with its base model embedded); None:
    # the directory loads as it is.
    config: Mapping[str, Any] | None = None
    backbone: str | None = None
    backbone_revision: str | None = None
    backbone_directory: Path | None = field(default=None, compare=False)


def _require_complete(
    what: str, model: str, revision: str | None, missing: Sequence[str], *, allow_download: bool
) -> None:
    """Refuse a model directory that lacks files its load needs (``missing``
    names the kinds of files). A cached snapshot can be incomplete (an
    interrupted download): with downloads off that is a model not cached."""
    if not missing:
        return
    kinds = ", ".join(missing)
    if is_local(model):
        raise _config_error(
            f"[detection.ner] {what} {model!r} is a local directory that lacks what"
            f" the loader needs: {kinds}"
        )
    if not allow_download:
        raise ModelNotCached(what, model, revision, tuple(missing))
    raise _config_error(
        f"[detection.ner] {what} {model!r} {_revision_text(revision)}: its repository"
        f" lacks what the loader needs: {kinds}"
    )


def _shards(directory: Path, index: str, *, what: str, model: str) -> list[str]:
    """The weight files an index file names, each a plain file name."""
    config = read_config(directory / index, what=what, model=model)
    weight_map = config.get("weight_map")
    names = set(weight_map.values()) if isinstance(weight_map, Mapping) else set()
    if not names or not all(
        isinstance(name, str) and name and "/" not in name and "\\" not in name for name in names
    ):
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {index} does not name its weight files"
        )
    return sorted(names)


def _hf_missing(
    directory: Path, names: Collection[str], *, model: str, extra_files: Sequence[str] = ()
) -> list[str]:
    """What an ``hf`` load needs that the files ``names`` lack (each of the
    catalog's ``extra_files`` by its name)."""
    missing = [] if "config.json" in names else ["config.json"]
    for single, index in (SAFETENSORS_FILES, PICKLE_FILES):
        if single in names:
            break
        if index in names:
            shards = _shards(directory, index, what="hf model", model=model)
            missing += [name for name in shards if name not in names]
            break
    else:
        missing.append("weights")
    if not has_vocabulary(names):
        missing.append("tokenizer files")
    missing += [name for name in extra_files if name not in names]
    return missing


def hf_files(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    allow_pickle_weights: bool,
    extra_files: Sequence[str] = (),
) -> ModelFiles:
    """The files an ``hf`` model loads from: its configuration, tokenizer
    and safetensors weights — or, only with ``allow_pickle_weights`` and
    only when it has no safetensors weights, its ``pytorch_model.bin`` —
    and the ``extra_files`` (exact names in the repository) the model
    catalog lists for it (``CatalogEntry.extra_files``: a BIOES/BILOU
    tagger's calibration file, which the build reads). A directory lacking
    any of the files a load needs — an extra file included — is refused
    (:func:`_require_complete`)."""
    what = "hf model"
    patterns = (*HF_PATTERNS, *extra_files)
    path = resolve_model(
        model,
        what=what,
        revision=revision,
        allow_download=allow_download,
        allow_patterns=patterns,
    )
    check_configs(path, CODE_CONFIG_FILES, what=what, model=model)
    if not has_files(path, SAFETENSORS_FILES):
        if not allow_pickle_weights:
            raise _config_error(
                f"[detection.ner] hf model {model!r} has no safetensors weights;"
                " set allow_pickle_weights = true to load pytorch_model.bin"
            )
        patterns = (*patterns, *HF_PICKLE_PATTERNS)
        if not is_local(model):
            # The repository lists no safetensors file (else the first
            # lookup would have needed it): fetch the pickle too.
            path = resolve_model(
                model,
                what=what,
                revision=revision,
                allow_download=allow_download,
                allow_patterns=patterns,
            )
    files = matching_files(path, patterns)
    missing = _hf_missing(path, files, model=model, extra_files=extra_files)
    _require_complete(what, model, revision, missing, allow_download=allow_download)
    return ModelFiles(model, path, files)


def hf_model_dir(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    allow_pickle_weights: bool,
    extra_files: Sequence[str] = (),
) -> Path:
    """The local directory an ``hf`` model loads from (:func:`hf_files`)."""
    return hf_files(
        model,
        revision=revision,
        allow_download=allow_download,
        allow_pickle_weights=allow_pickle_weights,
        extra_files=extra_files,
    ).directory


# --- GLiNER --------------------------------------------------------------------

GLINER_CONFIG = "gliner_config.json"
# A GLiNER checkpoint: its configuration, safetensors weights and (when
# it ships one) its tokenizer; pytorch_model.bin only when it has no
# safetensors (GLiNER loads a .bin with torch's weights_only loader).
GLINER_PATTERNS = (GLINER_CONFIG, "model.safetensors", *TOKENIZER_FILES)
GLINER_PICKLE = "pytorch_model.bin"
GLINER_WEIGHTS = ("model.safetensors", GLINER_PICKLE)
# What a GLiNER checkpoint's base model contributes: its configuration
# (embedded as encoder_config; it also tells transformers which tokenizer
# class to build) and its tokenizer. Never its weights: the checkpoint
# holds the encoder's.
BACKBONE_PATTERNS = ("config.json", *TOKENIZER_FILES)
# GLiNER's sub-configurations transformers builds a model from, and the
# model type it assumes when one names none (gliner/config.py).
_ENCODER_CONFIGS = ("encoder_config", "labels_encoder_config", "labels_decoder_config")
_DEFAULT_ENCODER_TYPE = "deberta-v2"
# The model_name of an assembled folder's configuration: a local path, so
# anything that still reads it reads files, never the Hub; and no absolute
# path, so the folder can be carried elsewhere.
LOCAL_MODEL_NAME = "."

logger = logging.getLogger("llm_redact")


def models_dir() -> Path:
    """Where llm-redact assembles model folders: ``$XDG_DATA_HOME/llm-redact/
    models`` (an empty XDG_DATA_HOME counts as unset)."""
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "llm-redact" / "models"


def _known_model_types(backend: str) -> Callable[[str], bool]:
    try:
        from transformers import CONFIG_MAPPING
    except ImportError as exc:
        raise _config_error(
            f'[detection.ner] backend = "{backend}" needs transformers, which the {backend}'
            f" extra installs; install it: uv sync --extra {backend}"
        ) from exc
    return lambda model_type: model_type in CONFIG_MAPPING


def _require_known_type(
    config: Mapping[str, Any], *, default: str | None, what: str, model: str, name: str
) -> None:
    model_type = config.get("model_type", default)
    backend = what.split()[0]  # "gliner model", "gliner base model", "gliner2 model"
    if not isinstance(model_type, str) or not _known_model_types(backend)(model_type):
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {name} names a model type transformers"
            " does not know; llm-redact never runs model code"
        )


def _identify(model: str, *, what: str = "gliner model") -> ModelIdentity | None:
    """Which model the GLiNER or GLiNER2 ``model`` value is
    (model_catalog.identify); a folder whose sidecar file cannot be read is
    a configuration error (``what`` names the model in it)."""
    from llm_redact.detection.model_catalog import SidecarError, identify

    try:
        return identify(model)
    except SidecarError as exc:
        raise _config_error(f"[detection.ner] {what}: {exc}") from exc


def _backbone_revision(model: str, backbone: str) -> str | None:
    """The catalog's pin of ``backbone`` for the GLiNER ``model`` (a Hub id,
    or a folder its sidecar names), or None — always None when ``backbone``
    names a local directory, even one named like the catalogued base model:
    a catalog pin names a Hub snapshot, never a folder's content (as
    ``NerConfig.revision_for`` for the model itself)."""
    from llm_redact.detection.model_catalog import lookup

    if is_local(backbone):
        return None
    identity = _identify(model)
    entry = lookup(identity.model_id) if identity is not None else None
    if entry is None or entry.backbone is None:
        return None
    if entry.backbone.casefold() != backbone.casefold():
        return None
    return entry.backbone_revision


def gliner_files(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    onnx_file: str | None = None,
    check_types: bool = True,
) -> ModelFiles:
    """The files a GLiNER model loads from with no network access.

    A self-contained checkpoint (its own tokenizer and an ``encoder_config``
    in ``gliner_config.json``) loads from its own directory. Any other needs
    an ASSEMBLED folder: its weights, its base model's tokenizer and
    ``config.json``, and a ``gliner_config.json`` (``ModelFiles.config``)
    that embeds the base model's configuration as ``encoder_config`` and
    names :data:`LOCAL_MODEL_NAME` — so GLiNER never asks the Hub for the
    base model's tokenizer or configuration, which it would at every load,
    at no fixed revision. The base model resolves at the catalog's pin.
    ``check_types`` (the loader) also refuses a model type transformers
    does not know, which imports transformers; ``llm-redact models`` and
    doctor check files only."""
    what = "gliner model"
    patterns: tuple[str, ...] = GLINER_PATTERNS
    if onnx_file is not None:
        # ONNX weights replace the torch ones: neither is fetched.
        patterns = (*(p for p in patterns if p not in GLINER_WEIGHTS), onnx_file)
    path = resolve_model(
        model, what=what, revision=revision, allow_download=allow_download, allow_patterns=patterns
    )
    if onnx_file is None and not (path / "model.safetensors").is_file():
        # No safetensors: GLiNER's weights_only .bin (fetched too from a
        # repository that lists no safetensors file).
        patterns = (*patterns, GLINER_PICKLE)
        if not is_local(model):
            path = resolve_model(
                model,
                what=what,
                revision=revision,
                allow_download=allow_download,
                allow_patterns=patterns,
            )
    if onnx_file is not None and not (path / onnx_file).is_file():
        raise _config_error(
            f"[detection.ner] gliner model {model!r} {_revision_text(revision)} has no ONNX"
            f" file {onnx_file!r} ([detection.ner.onnx] gliner)"
        )
    if is_local(model):
        _identify(model)  # a folder whose sidecar cannot be read is refused
    configs = check_configs(path, (GLINER_CONFIG, "tokenizer_config.json"), what=what, model=model)
    config = configs.get(GLINER_CONFIG)
    if config is None:
        raise _config_error(f"[detection.ner] gliner model {model!r} has no {GLINER_CONFIG}")
    if check_types:
        for key in _ENCODER_CONFIGS:
            sub = config.get(key)
            if isinstance(sub, Mapping):
                default = _DEFAULT_ENCODER_TYPE if key == "encoder_config" else None
                _require_known_type(
                    sub, default=default, what=what, model=model, name=GLINER_CONFIG
                )
    own = matching_files(path, (p for p in patterns if p != onnx_file))
    if onnx_file is not None:
        own[onnx_file] = path / onnx_file
    elif not any(name in own for name in GLINER_WEIGHTS):
        _require_complete(what, model, revision, ["weights"], allow_download=allow_download)
    has_tokenizer = "tokenizer_config.json" in own
    if has_tokenizer and isinstance(config.get("encoder_config"), Mapping):
        if not has_vocabulary(own):
            _require_complete(
                what, model, revision, ["tokenizer files"], allow_download=allow_download
            )
        return ModelFiles(model, path, own)
    backbone = config.get("model_name")
    if not isinstance(backbone, str) or not backbone:
        raise _config_error(
            f"[detection.ner] gliner model {model!r}: {GLINER_CONFIG} names no base model"
            " (model_name) for its tokenizer and encoder configuration"
        )
    backbone_revision = _backbone_revision(model, backbone)
    base_what = "gliner base model"
    base = resolve_model(
        backbone,
        what=base_what,
        revision=backbone_revision,
        allow_download=allow_download,
        allow_patterns=BACKBONE_PATTERNS,
    )
    base_configs = check_configs(base, CODE_CONFIG_FILES, what=base_what, model=backbone)
    base_config = base_configs.get("config.json")
    if base_config is None:
        raise _config_error(f"[detection.ner] {base_what} {backbone!r} has no config.json")
    if check_types:
        _require_known_type(
            base_config, default=None, what=base_what, model=backbone, name="config.json"
        )
    rewritten = dict(config)
    rewritten["model_name"] = LOCAL_MODEL_NAME
    if not isinstance(config.get("encoder_config"), Mapping):
        encoder = dict(base_config)
        # What GLiNER does to the configuration it fetches itself.
        vocab_size = config.get("vocab_size", -1)
        if vocab_size != -1:
            encoder["vocab_size"] = vocab_size
        rewritten["encoder_config"] = encoder
    # The checkpoint's weights (and its own tokenizer, when it ships one),
    # the base model's configuration (transformers' tokenizer-class lookup)
    # and, otherwise, the base model's tokenizer.
    files = {
        name: source
        for name, source in own.items()
        if name != GLINER_CONFIG and (has_tokenizer or name not in TOKENIZER_FILES)
    }
    base_files = matching_files(base, ("config.json", *(() if has_tokenizer else TOKENIZER_FILES)))
    files.update(base_files)
    if "tokenizer_config.json" not in files or not has_vocabulary(files):
        if has_tokenizer:
            _require_complete(
                what, model, revision, ["tokenizer files"], allow_download=allow_download
            )
        _require_complete(
            base_what,
            backbone,
            backbone_revision,
            ["tokenizer files"],
            allow_download=allow_download,
        )
    return ModelFiles(
        model,
        path,
        files,
        config=rewritten,
        backbone=backbone,
        backbone_revision=backbone_revision,
        backbone_directory=base,
    )


def gliner_model_dir(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    onnx_file: str | None = None,
) -> Path:
    """The local folder a GLiNER model loads from with no network access
    (:func:`gliner_files`): its own directory when it is self-contained,
    else a folder assembled under :func:`models_dir` (a startup WARNING
    names a base model the catalog does not pin)."""
    layout = gliner_files(
        model, revision=revision, allow_download=allow_download, onnx_file=onnx_file
    )
    if layout.config is None:
        return layout.directory
    backbone = layout.backbone or ""
    if layout.backbone_revision is None and not is_local(backbone):
        logger.warning(
            "[detection.ner] gliner model %r ships no tokenizer or encoder_config, and the"
            " model catalog pins no revision of its base model %r: the newest cached"
            " revision of its default branch loads",
            model,
            backbone,
        )
    return assemble_folder(model, layout.files, layout.config)


# --- GLiNER2 ---------------------------------------------------------------------

# A GLiNER2 checkpoint (the gliner2 package) is self-contained: its
# configuration, its encoder's configuration in a subfolder (gliner2 builds
# the encoder from it, never from the base model it names), its tokenizer
# and its weights. pytorch_model.bin only when it has no safetensors
# (gliner2 loads a .bin with torch's weights_only loader).
GLINER2_ENCODER_CONFIG = "encoder_config/config.json"
GLINER2_CONFIGS = ("config.json", GLINER2_ENCODER_CONFIG, "tokenizer_config.json")
GLINER2_PATTERNS = ("config.json", GLINER2_ENCODER_CONFIG, "model.safetensors", *TOKENIZER_FILES)


def gliner2_files(
    model: str, *, revision: str | None, allow_download: bool, check_types: bool = True
) -> ModelFiles:
    """The files a GLiNER2 model loads from with no network access: the
    checkpoint at its pinned revision, which must ship its configuration,
    its encoder configuration (a model type transformers knows: gliner2
    builds the encoder with ``trust_remote_code=True``), its tokenizer and
    its weights, none of the configurations naming code to import. A
    checkpoint is self-contained (no base model, nothing assembled), so
    ``ModelFiles.files`` holds ``encoder_config/config.json`` under its own
    subfolder. ``check_types`` (the loader) also refuses an encoder type
    transformers does not know, which imports transformers; ``llm-redact
    models`` and doctor check files only."""
    what = "gliner2 model"
    patterns: tuple[str, ...] = GLINER2_PATTERNS
    path = resolve_model(
        model, what=what, revision=revision, allow_download=allow_download, allow_patterns=patterns
    )
    if not (path / "model.safetensors").is_file():
        # No safetensors: gliner2's weights_only .bin (fetched too from a
        # repository that lists no safetensors file).
        patterns = (*patterns, GLINER_PICKLE)
        if not is_local(model):
            path = resolve_model(
                model,
                what=what,
                revision=revision,
                allow_download=allow_download,
                allow_patterns=patterns,
            )
    if is_local(model):
        # A folder whose sidecar cannot be read is refused, as gliner's and
        # hf's are (doctor and `llm-redact models` report it the same way).
        _identify(model, what=what)
    configs = check_configs(path, GLINER2_CONFIGS, what=what, model=model)
    for name in GLINER2_CONFIGS:
        if name not in configs:
            raise _config_error(
                f"[detection.ner] gliner2 model {model!r} has no {name}; a GLiNER2 checkpoint"
                " ships its configuration, its encoder configuration and its tokenizer"
            )
    if check_types:
        _require_known_type(
            configs[GLINER2_ENCODER_CONFIG],
            default=None,
            what=what,
            model=model,
            name=GLINER2_ENCODER_CONFIG,
        )
    files = matching_files(path, (p for p in patterns if p != GLINER2_ENCODER_CONFIG))
    files[GLINER2_ENCODER_CONFIG] = path / GLINER2_ENCODER_CONFIG
    missing = [] if any(name in files for name in GLINER_WEIGHTS) else ["weights"]
    if not has_vocabulary(files):
        missing.append("tokenizer files")
    _require_complete(what, model, revision, missing, allow_download=allow_download)
    return ModelFiles(model, path, dict(sorted(files.items())))


def gliner2_model_dir(model: str, *, revision: str | None, allow_download: bool) -> Path:
    """The local folder a GLiNER2 model loads from with no network access
    (:func:`gliner2_files`): the checkpoint's own directory."""
    return gliner2_files(model, revision=revision, allow_download=allow_download).directory


def config_text(config: Mapping[str, Any]) -> str:
    """A configuration file's text as llm-redact writes it."""
    return json.dumps(config, indent=2) + "\n"


def assemble_folder(model: str, files: Mapping[str, Path], config: Mapping[str, Any]) -> Path:
    """A folder under :func:`models_dir` holding ``files`` (linked, else
    copied) and ``config`` as its ``gliner_config.json``. Its name carries
    a digest of everything in it, so an existing folder of that name is
    complete by construction (it is built aside and renamed into place)."""
    text = config_text(config)
    digest = hashlib.sha256(text.encode("utf-8"))
    for name in sorted(files):
        digest.update(b"\0" + name.encode("utf-8") + b"\0")
        digest.update(os.path.realpath(files[name]).encode("utf-8"))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model.strip("/\\"))[-80:] or "model"
    root = models_dir() / "gliner"
    final = root / f"{slug}-{digest.hexdigest()[:16]}"
    if (final / GLINER_CONFIG).is_file():
        return final
    temp: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".assembling-", dir=root))
        for name, source in files.items():
            _link(source, temp / name)
        (temp / GLINER_CONFIG).write_text(text, encoding="utf-8")
        try:
            os.rename(temp, final)
            temp = None
        except OSError:
            if not (final / GLINER_CONFIG).is_file():  # not a concurrent twin
                raise
    except OSError as exc:
        raise _config_error(
            f"[detection.ner] gliner model {model!r}: cannot assemble its local folder"
            f" under {root} ({type(exc).__name__})"
        ) from exc
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    return final


def _link(source: Path, target: Path) -> None:
    """``target`` as a link to ``source``'s real file: a symbolic link,
    else a hard link, else a copy (Windows without the privilege)."""
    real = os.path.realpath(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(real, target)
        return
    except OSError:
        pass
    try:
        os.link(real, target)
    except OSError:
        shutil.copyfile(real, target)
