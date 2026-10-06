"""The student-model recipe (plan T43): a SKELETON that plans a run and trains nothing.

    uv run python scripts/pii_corpus/train_student.py plan --name acme-pii-student \\
        --base microsoft/deberta-v3-small --sources nemotron,privy,kiji --out ~/student-run

``plan`` checks every requested data source against the data manifest
(training_sources.toml): a source not listed, or one marked evaluation-only,
is refused; OpenPII 1.5M is refused unless ``--openpii-confirmation REF``
names AI4Privacy's written confirmation (plan D9 — set it only after the
owner has recorded that confirmation; REF goes into the manifest and the
model card); the private agent corpus must be a VERIFIED training share,
never the frozen ``agent-eval`` set. It checks the base model (an Apache-2.0
or MIT encoder, or a catalogued GLiNER-PII checkpoint that is not
restricted), then writes, into a directory outside every git work tree:

* ``data-manifest.json``: every source with its license, attribution,
  revision, lineage and split, the base model with its license, revision and
  lineage, the OpenPII confirmation reference when given, and the recorded
  hyperparameters; no text;
* ``MODEL_CARD.md``: MODEL_CARD_TEMPLATE.md filled from the manifest, with
  the bench results left to fill in after the evaluation.

``train`` is a deliberate stub: training is the step TRAINING.md describes
and an owner decision; this skeleton never downloads data or trains.
"""

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_redact.bench.datasets.agent_eval import FORMAT as FROZEN_FORMAT  # noqa: E402
from llm_redact.bench.datasets.agent_eval import file_sha256, manifest_path  # noqa: E402
from llm_redact.detection.model_catalog import lookup  # noqa: E402
from pii_corpus.private_files import (  # noqa: E402
    CorpusError,
    output_problem,
    read_jsonl,
    write_json,
)

HERE = Path(__file__).resolve().parent
SOURCES_FILE = HERE / "training_sources.toml"
TEMPLATE_FILE = HERE / "MODEL_CARD_TEMPLATE.md"
MANIFEST_FORMAT = "llm-redact-student-data/1"
OPENPII = "openpii"
AGENT_CORPUS = "agent-corpus"
# Plain encoders a token-classification student may start from (checked
# 2026-10-05: license tag and main commit on the Hugging Face Hub).
ENCODERS: Mapping[str, tuple[str, str]] = {
    "microsoft/deberta-v3-small": ("MIT", "a36c739020e01763fe789b4b85e2df55d6180012"),
    "microsoft/deberta-v3-base": ("MIT", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"),
}
# Recorded with every plan; TRAINING.md explains them.
HYPERPARAMETERS: Mapping[str, Any] = {
    "seed": 42,
    "epochs": 3,
    "learning_rate": 3e-5,
    "warmup_ratio": 0.1,
    "batch_size": 16,
    "max_length_tokens": 512,
    "label_set": "canonical NER types + EMAIL, PHONE (BIO tags for hf; type prompts for gliner)",
}
NOT_IMPLEMENTED = (
    "training is not implemented: this is the recipe skeleton (scripts/pii_corpus/TRAINING.md"
    " describes the run); nothing was trained"
)


def load_sources(path: Path = SOURCES_FILE) -> dict[str, dict[str, Any]]:
    try:
        sources = tomllib.loads(path.read_text(encoding="utf-8")).get("sources")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CorpusError(f"cannot read {path.name}: {type(exc).__name__}") from exc
    if not isinstance(sources, dict):
        raise CorpusError(f"{path.name} holds no [sources] table")
    return sources


def base_model(model_id: str) -> dict[str, Any]:
    """The base model's facts, or CorpusError naming why it may not be one."""
    if model_id in ENCODERS:
        license_id, revision = ENCODERS[model_id]
        return {
            "id": model_id,
            "backend": "hf",
            "license": license_id,
            "revision": revision,
            "lineage": [],
        }
    entry = lookup(model_id)
    if entry is None or "gliner" not in entry.backends or entry.prefix:
        known = ", ".join(sorted(ENCODERS))
        raise CorpusError(
            f"base model {model_id}: not an allowed encoder ({known}) or a catalogued GLiNER model"
        )
    if entry.status == "restricted" or entry.license not in ("Apache-2.0", "MIT"):
        raise CorpusError(f"base model {model_id}: catalog status {entry.status}, {entry.license}")
    if entry.revision is None:
        raise CorpusError(f"base model {model_id}: the catalog records no revision pin")
    return {
        "id": model_id,
        "backend": "gliner",
        "license": entry.license,
        "revision": entry.revision,
        "lineage": list(entry.lineage),
    }


def _agent_corpus(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise CorpusError("source agent-corpus needs --agent-corpus PATH (a review.py output)")
    try:
        frozen = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        frozen = None
    if isinstance(frozen, dict) and frozen.get("format") == FROZEN_FORMAT:
        raise CorpusError(
            f"{path.name} is the frozen agent-eval set: it is evaluation-only and must never be"
            " trained on (use a separately verified training share)"
        )
    rows = 0
    for number, row in read_jsonl(path):
        if not isinstance(row.get("review"), dict):
            raise CorpusError(f"{path.name} line {number}: an unverified row (no review record)")
        rows += 1
    return {"rows": rows, "sha256": file_sha256(path), "file": path.name}


def plan_sources(
    names: Sequence[str],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    openpii_confirmation: str | None,
    agent_corpus: Path | None,
) -> list[dict[str, Any]]:
    """The manifest entries of the requested sources; CorpusError for the
    first one the recipe may not train on."""
    if not names:
        raise CorpusError("name at least one source (--sources)")
    planned = []
    for name in names:
        entry = sources.get(name)
        if entry is None:
            raise CorpusError(
                f"source {name!r} is not in the data manifest (training_sources.toml)"
            )
        training = entry.get("training")
        if training == "refused":
            raise CorpusError(f"source {name!r} is evaluation-only: {entry.get('reason')}")
        if training == "needs-confirmation":
            if name != OPENPII or not (openpii_confirmation or "").strip():
                raise CorpusError(
                    f"source {name!r} is refused for training until {entry.get('gate')} is"
                    " recorded: pass --openpii-confirmation REF only once it is (TRAINING.md)"
                )
        elif training != "allowed":
            raise CorpusError(f"source {name!r} has no valid training status in the manifest")
        record = {"name": name, **{k: v for k, v in entry.items() if k != "training"}}
        if name == AGENT_CORPUS:
            record.update(_agent_corpus(agent_corpus))
        if name == OPENPII:
            record["confirmation"] = (openpii_confirmation or "").strip()
        planned.append(record)
    return planned


def render_card(template: str, manifest: Mapping[str, Any]) -> str:
    base = manifest["base_model"]
    rows = [
        f"| {s['name']} | {s.get('hub_id', 'local')} | {s.get('revision', '-')} |"
        f" {s['license']} | {s['lineage']} |"
        for s in manifest["sources"]
    ]
    confirmation = next(
        (s["confirmation"] for s in manifest["sources"] if s["name"] == OPENPII),
        "not used",
    )
    values = {
        "model_name": manifest["name"],
        "date": manifest["planned"],
        "base_model": base["id"],
        "base_backend": base["backend"],
        "base_license": base["license"],
        "base_revision": base["revision"],
        "base_lineage": ", ".join(base["lineage"]) or "none recorded",
        "training_data_rows": "\n".join(rows),
        "attributions": "\n".join(f"- {s['attribution']}" for s in manifest["sources"]),
        "openpii_confirmation": confirmation,
        "hyperparameters": "\n".join(
            f"- `{k}`: {v}" for k, v in manifest["hyperparameters"].items()
        ),
    }
    card = template
    for key, value in values.items():
        card = card.replace("{{" + key + "}}", str(value))
    return card


def plan(args: argparse.Namespace, today: str) -> dict[str, Any]:
    problem = output_problem(args.out / "data-manifest.json")
    if problem is not None:
        raise CorpusError(problem)
    if args.out.exists() and not args.out.is_dir():
        raise CorpusError(f"{args.out} is not a directory")
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise CorpusError(f"{args.out} is not empty; pass --force to plan into it")
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "name": args.name,
        "planned": today,
        "base_model": base_model(args.base),
        "sources": plan_sources(
            names,
            load_sources(),
            openpii_confirmation=args.openpii_confirmation,
            agent_corpus=args.agent_corpus,
        ),
        "hyperparameters": dict(HYPERPARAMETERS),
        "evaluation_only": ["the agent-eval frozen set", "the bench's test/validation splits"],
    }
    args.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json(args.out / "data-manifest.json", manifest, overwrite=True)
    card = render_card(TEMPLATE_FILE.read_text(encoding="utf-8"), manifest)
    (args.out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/pii_corpus/train_student.py",
        description="Plan a student-model run on license-clean data (a skeleton: trains nothing).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    planned = commands.add_parser("plan", help="check sources and write the manifest and card")
    planned.add_argument("--name", required=True, help="the student model's name")
    planned.add_argument("--base", required=True, help="base model id (see TRAINING.md)")
    planned.add_argument("--sources", required=True, help="comma-separated source names")
    planned.add_argument("--agent-corpus", type=Path, help="a verified training share (JSONL)")
    planned.add_argument(
        "--openpii-confirmation",
        metavar="REF",
        help="the reference of AI4Privacy's written confirmation (plan D9); without it the"
        " openpii source is refused",
    )
    planned.add_argument("--out", type=Path, required=True, help="the run directory")
    planned.add_argument("--force", action="store_true", help="plan into a non-empty directory")
    train = commands.add_parser("train", help="not implemented (the recipe skeleton)")
    train.add_argument("run", type=Path, help="a planned run directory")
    return parser


def main(argv: Sequence[str] | None = None, *, today: str | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        print(f"error: {NOT_IMPLEMENTED}", file=sys.stderr)
        return 2
    try:
        manifest = plan(args, today or datetime.now(UTC).date().isoformat())
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    names = ", ".join(s["name"] for s in manifest["sources"])
    print(f"{args.out}: planned {args.name} on {manifest['base_model']['id']} from {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
