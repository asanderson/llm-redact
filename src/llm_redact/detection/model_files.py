"""Where NER model files come from: local, pinned, and free of model code.

The ``hf`` and ``gliner`` backends load Hugging Face Hub models. This module
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
  GLiNER never fetches the base model from the Hub at load time.

Messages name the backend, the model id and the revision only.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_redact.detection.model_catalog import MODEL_ID_RE, pinned_revision

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
    from llm_redact.config import ConfigError

    return ConfigError(message)


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
    A local directory is returned as it is; a configured ``revision`` for
    one (any but the catalog's own pin for the same name) is an error. A
    Hub model id is looked up at ``revision`` with ``allow_patterns``,
    from the local Hugging Face cache only unless ``allow_download``.
    """
    if is_local(model):
        if revision is not None and revision != pinned_revision(model):
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
            f"[detection.ner] {what} {model!r} needs huggingface_hub, which the hf and"
            " gliner extras install; install the backend's extra"
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
            raise _config_error(
                f"[detection.ner] {what} {model!r} {_revision_text(revision)} is not"
                " (completely) in the local Hugging Face cache, and downloads are off"
                " (an older revision in the cache does not count); run"
                " `llm-redact models pull`, or set [detection.ner] allow_download = true"
                " to fetch it at startup (a reload never downloads)"
            ) from exc
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


def hf_model_dir(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    allow_pickle_weights: bool,
    extra_files: Sequence[str] = (),
) -> Path:
    """The local directory an ``hf`` model loads from: its configuration,
    tokenizer and safetensors weights — or, only with
    ``allow_pickle_weights`` and only when it has no safetensors weights,
    its ``pytorch_model.bin`` — and the ``extra_files`` (exact names in the
    repository) the model catalog lists for it, such as a tagger's
    calibration file."""
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
    if has_files(path, SAFETENSORS_FILES):
        return path
    if not allow_pickle_weights:
        raise _config_error(
            f"[detection.ner] hf model {model!r} has no safetensors weights;"
            " set allow_pickle_weights = true to load pytorch_model.bin"
        )
    if is_local(model):
        return path
    # The repository lists no safetensors file (else the first lookup would
    # have needed it): fetch the pickle too.
    return resolve_model(
        model,
        what=what,
        revision=revision,
        allow_download=allow_download,
        allow_patterns=(*patterns, *HF_PICKLE_PATTERNS),
    )


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


def _known_model_types() -> Callable[[str], bool]:
    try:
        from transformers import CONFIG_MAPPING
    except ImportError as exc:
        raise _config_error(
            '[detection.ner] backend = "gliner" needs transformers, which the gliner'
            " extra installs; install it: uv sync --extra gliner"
        ) from exc
    return lambda model_type: model_type in CONFIG_MAPPING


def _require_known_type(
    config: Mapping[str, Any], *, default: str | None, what: str, model: str, name: str
) -> None:
    model_type = config.get("model_type", default)
    if not isinstance(model_type, str) or not _known_model_types()(model_type):
        raise _config_error(
            f"[detection.ner] {what} {model!r}: {name} names a model type transformers"
            " does not know; llm-redact never runs model code"
        )


def _backbone_revision(model: str, backbone: str) -> str | None:
    """The catalog's pin of ``backbone`` for the GLiNER ``model`` (a Hub id,
    or a folder its sidecar names), or None."""
    from llm_redact.detection.model_catalog import SidecarError, identify, lookup

    try:
        identity = identify(model)
    except SidecarError as exc:
        raise _config_error(f"[detection.ner] gliner model: {exc}") from exc
    entry = lookup(identity.model_id) if identity is not None else None
    if entry is None or entry.backbone is None:
        return None
    if entry.backbone.casefold() != backbone.casefold():
        return None
    return entry.backbone_revision


def gliner_model_dir(
    model: str,
    *,
    revision: str | None,
    allow_download: bool,
    onnx_file: str | None = None,
) -> Path:
    """The local folder a GLiNER model loads from with no network access.

    A self-contained checkpoint (its own tokenizer and an ``encoder_config``
    in ``gliner_config.json``) is its own folder. Any other is assembled
    under :func:`models_dir`: links to its weights, its base model's
    tokenizer and ``config.json``, and a ``gliner_config.json`` that embeds
    the base model's configuration as ``encoder_config`` and names
    :data:`LOCAL_MODEL_NAME` — so GLiNER never asks the Hub for the base
    model's tokenizer or configuration, which it would at every load, at
    no fixed revision. The base model is resolved at the catalog's pin
    (a startup WARNING names one the catalog does not pin)."""
    what = "gliner model"
    patterns: tuple[str, ...] = GLINER_PATTERNS
    if onnx_file is not None:
        # ONNX weights replace the torch ones: neither is fetched.
        patterns = (*(p for p in patterns if p not in GLINER_WEIGHTS), onnx_file)
    path = resolve_model(
        model, what=what, revision=revision, allow_download=allow_download, allow_patterns=patterns
    )
    if onnx_file is None and not (path / "model.safetensors").is_file() and not is_local(model):
        # No safetensors in the repository: GLiNER's weights_only .bin.
        path = resolve_model(
            model,
            what=what,
            revision=revision,
            allow_download=allow_download,
            allow_patterns=(*patterns, GLINER_PICKLE),
        )
    if onnx_file is not None and not (path / onnx_file).is_file():
        raise _config_error(
            f"[detection.ner] gliner model {model!r} {_revision_text(revision)} has no ONNX"
            f" file {onnx_file!r} ([detection.ner.onnx] gliner)"
        )
    configs = check_configs(path, (GLINER_CONFIG, "tokenizer_config.json"), what=what, model=model)
    config = configs.get(GLINER_CONFIG)
    if config is None:
        raise _config_error(f"[detection.ner] gliner model {model!r} has no {GLINER_CONFIG}")
    for key in _ENCODER_CONFIGS:
        sub = config.get(key)
        if isinstance(sub, Mapping):
            default = _DEFAULT_ENCODER_TYPE if key == "encoder_config" else None
            _require_known_type(sub, default=default, what=what, model=model, name=GLINER_CONFIG)
    has_tokenizer = (path / "tokenizer_config.json").is_file()
    if has_tokenizer and isinstance(config.get("encoder_config"), Mapping):
        return path
    backbone = config.get("model_name")
    if not isinstance(backbone, str) or not backbone:
        raise _config_error(
            f"[detection.ner] gliner model {model!r}: {GLINER_CONFIG} names no base model"
            " (model_name) for its tokenizer and encoder configuration"
        )
    backbone_revision = _backbone_revision(model, backbone)
    if backbone_revision is None and not is_local(backbone):
        logger.warning(
            "[detection.ner] gliner model %r ships no tokenizer or encoder_config, and the"
            " model catalog pins no revision of its base model %r: the newest cached"
            " revision of its default branch loads",
            model,
            backbone,
        )
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
    files: dict[str, Path] = {}
    names = [*GLINER_WEIGHTS, *([onnx_file] if onnx_file is not None else [])]
    if has_tokenizer:
        names += list(TOKENIZER_FILES)
    for name in names:
        if (path / name).is_file():
            files[name] = path / name
    tokenizer_source = () if has_tokenizer else TOKENIZER_FILES
    for name in ("config.json", *tokenizer_source):
        if (base / name).is_file():
            files[name] = base / name
    return assemble_folder(model, files, rewritten)


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
