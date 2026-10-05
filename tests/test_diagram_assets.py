"""The committed diagram renders: every Mermaid source has its PNG, and every
source that opts in with ``%% animate`` has an animated SVG (CSS-only, with a
reduced-motion fallback) and a GIF beside it — scripts/animate_diagrams.py
output, pinned so a source edit cannot ship without its renders."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DIAGRAMS = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
SOURCES = sorted(DIAGRAMS.glob("*.mmd"))
ANIMATED = [s for s in SOURCES if re.search(r"^%%\s*animate\b", s.read_text(), re.MULTILINE)]


def test_sources_exist() -> None:
    assert {s.stem for s in SOURCES} >= {
        "architecture",
        "ner-pipeline",
        "ner-prefetch",
        "security-gates",
        "sequence-chat",
    }
    assert {s.stem for s in ANIMATED} == {
        "architecture",
        "ner-prefetch",
        "security-gates",
        "sequence-chat",
        "sequence-streaming",
    }


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.stem)
def test_every_source_has_its_png(source: Path) -> None:
    png = source.with_suffix(".png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("source", ANIMATED, ids=lambda s: s.stem)
def test_animated_svg_and_gif(source: Path) -> None:
    svg = source.with_suffix(".svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "@keyframes" in svg
    assert "<script" not in svg  # CSS only: plays as an <img>, nothing executes
    assert "prefers-reduced-motion" in svg
    assert ".anim-flow" in svg or "llmr-seq-0" in svg
    gif = source.with_suffix(".gif").read_bytes()
    assert gif[:6] in (b"GIF89a", b"GIF87a")
    assert b"NETSCAPE2.0" in gif[:1024]  # loops


@pytest.mark.parametrize("source", [s for s in SOURCES if s not in ANIMATED], ids=lambda s: s.stem)
def test_static_diagrams_have_no_animated_render(source: Path) -> None:
    assert not source.with_suffix(".svg").exists()
    assert not source.with_suffix(".gif").exists()
