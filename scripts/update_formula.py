#!/usr/bin/env python3
"""Point Formula/llm-redact.rb at a released sdist on PyPI. Stdlib only.

Run after a release publishes to PyPI (see docs/RELEASING.md):

    python scripts/update_formula.py 0.6.0

Rewrites only the top-level url/sha256 lines (the resource stanzas pin the
runtime dependency closure and change only when uv.lock does).
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

FORMULA = Path(__file__).resolve().parent.parent / "Formula" / "llm-redact.rb"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    version = sys.argv[1].lstrip("v")
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/llm-redact-proxy/{version}/json", timeout=30
    ) as response:
        data = json.load(response)
    sdist = next(u for u in data["urls"] if u["packagetype"] == "sdist")

    text = FORMULA.read_text()
    text, url_count = re.subn(r'^  url ".*"$', f'  url "{sdist["url"]}"', text, flags=re.M)
    text, sha_count = re.subn(
        r'^  sha256 ".*"$', f'  sha256 "{sdist["digests"]["sha256"]}"', text, flags=re.M
    )
    if url_count != 1 or sha_count != 1:
        print(f"expected exactly one top-level url/sha256 line, found {url_count}/{sha_count}")
        return 1
    FORMULA.write_text(text)
    print(f"pinned Formula/llm-redact.rb to llm-redact-proxy {version}")
    print(f"  {sdist['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
