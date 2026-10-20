"""A bidirectional diagram link must show a complete request/response loop."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "animate_diagrams.py"
_spec = importlib.util.spec_from_file_location("animate_diagrams", _SCRIPT)
assert _spec is not None and _spec.loader is not None
animator = importlib.util.module_from_spec(_spec)
sys.modules["animate_diagrams"] = animator
_spec.loader.exec_module(animator)


@pytest.mark.parametrize("thick", [False, True], ids=["normal", "thick"])
@pytest.mark.parametrize("bidirectional", [False, True], ids=["one-way", "round-trip"])
def test_flow_animation_covers_every_direction_in_its_gif_period(
    thick: bool, bidirectional: bool
) -> None:
    # Hand-authored Mermaid-shaped output, independent of the animator.
    thickness = "thick" if thick else "normal"
    start_marker = ' marker-start="url(#arrow)"' if bidirectional else ""
    svg = (
        "<svg><style></style>"
        '<g class="node default" id="diagram-flowchart-client-0"/>'
        '<g class="node default" id="diagram-flowchart-proxy-1"/>'
        '<path id="diagram-L_client_proxy_0" d="M0,0 L20,0" '
        f'class="edge-thickness-{thickness} flowchart-link"'
        f'{start_marker} marker-end="url(#arrow)"/>'
        "</svg>"
    )

    animated, period = animator.animate_flowchart(svg)

    # Capturing only one forward period would omit the restored response.
    assert period == animator.FLOW_PERIOD_S * (2 if bidirectional else 1)
    paths = ET.fromstring(animated).findall("path")
    original, overlay = paths
    assert original.attrib["marker-end"] == "url(#arrow)"
    assert ("marker-start" in original.attrib) is bidirectional
    assert overlay.attrib["d"] == original.attrib["d"]
    classes = overlay.attrib["class"].split()
    assert "anim-flow" in classes
    assert ("anim-thick" in classes) is thick
    assert ("anim-roundtrip" in classes) is bidirectional
    if bidirectional:
        assert ".anim-flow.anim-roundtrip{animation-direction:alternate}" in animated
    assert "prefers-reduced-motion" in animated
