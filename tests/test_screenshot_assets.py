"""The committed screenshot SVGs must be well-formed XML.

SVG is XML: one unescaped quote inside an attribute makes the whole file
malformed, and GitHub/browsers render it as a broken image while the
file itself still "exists" (the preview.svg regression — its aria-label
embedded the `--text "…"` command verbatim). The renderer is exercised
directly with a quote-bearing command so the generator can't regress,
and every committed SVG is parsed so a bad re-capture can't land.
"""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = ROOT / "scripts" / "capture_plugin_screenshots.py"
_spec = importlib.util.spec_from_file_location("capture_plugin_screenshots", _SCRIPT)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
sys.modules["capture_plugin_screenshots"] = capture
_spec.loader.exec_module(capture)

_COMMITTED_SVGS = sorted((ROOT / "docs").rglob("*.svg"))


@pytest.mark.parametrize("svg", _COMMITTED_SVGS, ids=lambda p: p.name)
def test_committed_svg_is_well_formed_xml(svg: Path) -> None:
    ET.parse(svg)


def test_docs_actually_contain_svgs() -> None:
    # The parametrized test above is vacuous if the glob finds nothing.
    assert _COMMITTED_SVGS, "expected committed SVG screenshots under docs/"


def test_renderer_escapes_quotes_in_command() -> None:
    rendered = capture.render_terminal_svg(
        'llm-redact preview --text "quote & <angle> test"', "line 1\nline 2 <>&\"'"
    )
    root = ET.fromstring(rendered)
    assert 'preview --text "quote & <angle> test"' in root.attrib["aria-label"]
