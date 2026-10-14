"""Check, inside a built image, that the AGPL-3.0 text ships with it.

Two copies: where a reader looks for it (/usr/share/doc/llm-redact/LICENSE,
which the Dockerfile's runtime stage copies in) and in the installed
distribution's metadata (licenses/LICENSE, which the wheel built in the
builder stage carries), as the same text. scripts/container_smoke.sh runs it
in the stock image and ci.yml's container-ner job in the -ner image:

    python image_license_check.py

Exit 0 when both copies are there, 1 (naming what is wrong) otherwise.
"""

import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

DOC = Path("/usr/share/doc/llm-redact/LICENSE")
DISTRIBUTION = "llm-redact-proxy"
METADATA_FILE = "licenses/LICENSE"
MARKER = "GNU AFFERO GENERAL PUBLIC LICENSE"


def doc_problems(doc: str | None) -> list[str]:
    """What is wrong with the doc copy's text (None: the file is absent)."""
    if doc is None:
        return [f"{DOC} is missing"]
    if MARKER not in doc:
        return [f"{DOC} does not hold the AGPL-3.0 text"]
    return []


def problems(doc: str | None, metadata: str | None) -> list[str]:
    """What is wrong with the two copies' texts (None: the file is absent):
    each copy is checked for the text, and the metadata copy compared with
    the doc copy when both hold it."""
    found = doc_problems(doc)
    if metadata is None:
        found.append(f"the {DISTRIBUTION} distribution carries no {METADATA_FILE}")
    elif MARKER not in metadata:
        found.append(
            f"the {DISTRIBUTION} distribution's {METADATA_FILE} does not hold the AGPL-3.0 text"
        )
    elif not found and metadata != doc:
        found.append(f"the {DISTRIBUTION} distribution's {METADATA_FILE} is not {DOC}")
    return found


def main() -> int:
    doc = DOC.read_text(encoding="utf-8") if DOC.is_file() else None
    try:
        found = problems(doc, distribution(DISTRIBUTION).read_text(METADATA_FILE))
    except PackageNotFoundError:
        found = [*doc_problems(doc), f"the {DISTRIBUTION} distribution is not installed"]
    for problem in found:
        print(f"FAIL: {problem}", file=sys.stderr)
    if not found:
        print(f"the AGPL-3.0 text is at {DOC} and in the distribution's {METADATA_FILE}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
