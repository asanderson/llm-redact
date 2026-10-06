"""Which model each Hugging Face Hub NER backend loads, and what is known
about it, from the configuration and local files only.

Shared by doctor's ``models`` area, ``llm-redact models`` and (computed
once per detector build) ``/status`` and the startup warnings. Nothing here
loads a model or opens a network connection: a Hub model's facts come from
the configuration and the model catalog, a local folder's from its
``llm-redact-model.json`` sidecar, and :func:`local_files` asks the local
Hugging Face cache only (unless a caller that may download says so).
Messages and fields name backends, model ids, revisions and catalog facts,
never anything a model read.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_redact.detection.model_catalog import (
    DEFAULT_MODELS,
    HUB_BACKENDS,
    CatalogEntry,
    SidecarError,
    lookup,
    read_sidecar,
)

if TYPE_CHECKING:
    from llm_redact.detection.engine import NerConfig
    from llm_redact.detection.model_files import ModelFiles


@dataclass(frozen=True)
class ModelSource:
    """One Hub backend's model: what the configuration names, which model
    and commit that is, and its catalog entry."""

    backend: str
    # The configured value (or the backend's default): a Hub id or a folder.
    model: str
    local: bool
    # The Hub id: the configured one, or the one a folder's sidecar names
    # (None: a folder without a sidecar, which nothing identifies).
    model_id: str | None
    # The commit it loads; a folder's is its sidecar's (None: unpinned).
    revision: str | None
    # Where the revision comes from: "config" ([detection.ner.revisions]),
    # "catalog" (the model catalog's pin) or "sidecar" (a folder's file).
    pinned_by: str | None
    entry: CatalogEntry | None
    # A folder's sidecar that exists but cannot be read (the startup
    # refuses such a folder); the message names the file only.
    sidecar_problem: str | None = None

    @property
    def pinned(self) -> bool:
        return self.revision is not None

    def status_fields(self) -> dict[str, Any]:
        """The ``/status`` ``detection.ner.backends.<backend>`` fields."""
        return {
            "source": "local" if self.local else "hub",
            "model_id": self.model_id,
            "revision": self.revision,
            "pinned": self.pinned,
            "catalog": self.entry.status if self.entry is not None else None,
            "license": self.entry.license if self.entry is not None else None,
        }

    def restricted_warning(self) -> str | None:
        """The startup warning for a model the catalog lists as restricted
        (neutral facts, a link and the check date: owner decision D14), or
        None."""
        if self.entry is None or self.entry.status != "restricted":
            return None
        return (
            f"[detection.ner] {self.backend} model {self.model!r} has model catalog status"
            f' "restricted": {self.entry.describe()}'
        )


# /status fields of a backend whose model is not a Hub snapshot (spaCy,
# Presidio, Stanza) or that a plugin built.
UNKNOWN_SOURCE_FIELDS: dict[str, Any] = {
    "source": None,
    "model_id": None,
    "revision": None,
    "pinned": None,
    "catalog": None,
    "license": None,
}


def model_source(ner: "NerConfig", backend: str) -> ModelSource:
    """The model ``backend`` (``gliner`` or ``hf``) loads under ``ner``."""
    from llm_redact.detection.model_files import is_local

    model = ner.model_for(backend) or DEFAULT_MODELS[backend]
    if is_local(model):
        try:
            identity = read_sidecar(model)
        except SidecarError as exc:
            return ModelSource(backend, model, True, None, None, None, None, str(exc))
        if identity is None:
            return ModelSource(backend, model, True, None, None, None, None)
        return ModelSource(
            backend,
            model,
            True,
            identity.model_id,
            identity.revision,
            "sidecar" if identity.revision is not None else None,
            lookup(identity.model_id),
        )
    entry = lookup(model)
    configured = ner.configured_revision(backend)
    if configured is not None:
        return ModelSource(backend, model, False, model, configured, "config", entry)
    pin = entry.revision if entry is not None else None
    return ModelSource(backend, model, False, model, pin, "catalog" if pin else None, entry)


def hub_sources(ner: "NerConfig") -> list[ModelSource]:
    """The model of every active backend that loads Hugging Face Hub
    models, in configuration order (``enabled`` is not consulted)."""
    return [
        model_source(ner, backend) for backend in ner.active_backends() if backend in HUB_BACKENDS
    ]


def local_files(
    ner: "NerConfig",
    source: ModelSource,
    *,
    revision: str | None = None,
    allow_download: bool = False,
) -> "ModelFiles":
    """The files ``source``'s load reads, as the loader finds them — from
    the local Hugging Face cache or the folder, at the revision a build
    asks for (``revision`` overrides it; ``models pull`` passes the commit
    it resolved) — checked complete. Model types are not checked (that
    imports transformers); the loader does. Raises ConfigError
    (``ModelNotCached`` for a model the cache lacks)."""
    from llm_redact.detection.model_files import gliner_files, hf_files

    if revision is None:
        revision = ner.revision_for(source.backend)
    if source.backend == "hf":
        return hf_files(
            source.model,
            revision=revision,
            allow_download=allow_download,
            allow_pickle_weights=ner.allow_pickle_weights,
        )
    return gliner_files(
        source.model,
        revision=revision,
        allow_download=allow_download,
        onnx_file=ner.onnx_for("gliner"),
        check_types=False,
    )
