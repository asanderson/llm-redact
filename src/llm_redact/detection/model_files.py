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
  ``[detection.ner] allow_download`` is set;
* a model whose configuration names code to run (``auto_map``) is refused:
  llm-redact never loads with ``trust_remote_code``;
* ``hf`` weights must be safetensors unless ``allow_pickle_weights`` is set
  (owner decision D3): a ``pytorch_model.bin`` is a pickle, and a pickle can
  run code when it is loaded.

Messages name the backend, the model id and the revision only.
"""

from collections.abc import Iterable, Mapping, Sequence
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
                " to fetch it at startup"
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
) -> Path:
    """The local directory an ``hf`` model loads from: its configuration,
    tokenizer and safetensors weights — or, only with
    ``allow_pickle_weights`` and only when it has no safetensors weights,
    its ``pytorch_model.bin``."""
    what = "hf model"
    path = resolve_model(
        model,
        what=what,
        revision=revision,
        allow_download=allow_download,
        allow_patterns=HF_PATTERNS,
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
        allow_patterns=(*HF_PATTERNS, *HF_PICKLE_PATTERNS),
    )
