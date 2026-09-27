#!/usr/bin/env python3
"""Render animated versions of the Mermaid diagrams: an SVG whose data flow
moves, and a GIF of it for viewers that do not run SVG animations.

A diagram opts in with a ``%% animate`` comment in its ``.mmd`` source; the
others (structural boundary diagrams) stay static. For each opted-in
diagram this script renders the SVG with mermaid-cli (the same pinned
version and background as the PNGs), then rewrites it:

- flowcharts: every DIRECTED edge that does not end at a refusal node
  (Mermaid class ``refuse``) gets an overlay path with a travelling dash —
  the request/response flow — on top of the untouched original edge, so the
  static appearance is unchanged and the arrowheads stay;
- sequence diagrams: messages, their numbers, activations and notes appear
  in chronological order (top to bottom), each arrow drawing itself, then
  the finished diagram holds before the loop restarts.

Everything is CSS inside the SVG's own ``<style>``: no scripts, so the SVG
animates when embedded with ``<img>`` (GitHub, most Markdown viewers) and
degrades to the complete static picture under ``prefers-reduced-motion``.
The GIF is captured frame by frame from the same SVG in a headless Chromium
(Playwright, the CSS animations paused and seeked), scaled down and
assembled with Pillow. Path coordinates are rounded (mermaid emits ~15
decimals), which halves the SVG without a visible change.

Usage: scripts/render_diagrams.sh runs this after the PNGs; direct:
    uv run --no-project --with playwright --with pillow \
        python scripts/animate_diagrams.py [--no-gif] [--only NAME]
Environment: PUPPETEER_EXECUTABLE_PATH (Chromium for mermaid-cli AND the GIF
frames), MMDC_PUPPETEER_CONFIG (an existing puppeteer JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MERMAID_CLI = "@mermaid-js/mermaid-cli@11"  # pinned like render_diagrams.sh
ANIMATE_MARKER = re.compile(r"^%%\s*animate\b", re.MULTILINE)

# --- timing -----------------------------------------------------------------------
FLOW_PERIOD_S = 1.2  # one dash cycle; the GIF loops on it seamlessly
SEQ_STEP_S = 0.8  # one message per step
SEQ_HOLD_S = 2.4  # the finished diagram stays before the loop restarts
SEQ_FADE_S = 0.25
SEQ_DRAW_S = 0.45
GIF_FPS_FLOW = 12
GIF_FPS_SEQ = 10
GIF_MAX_WIDTH = 1100

ACCENT = "#0969da"  # the palette's gate blue

_TAG_RE = re.compile(r"<(g|line|path|rect|text)\b[^>]*?/?>")
_NUM_RE = re.compile(r"-?\d+\.\d{3,}")


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf'\b{name}="([^"]*)"', tag)
    return match.group(1) if match else None


def _classes(tag: str) -> set[str]:
    return set((_attr(tag, "class") or "").split())


def _round_numbers(svg: str) -> str:
    """Round long decimals in path data and coordinates (2 places is far below
    a device pixel at any sensible zoom)."""
    return _NUM_RE.sub(lambda m: f"{float(m.group(0)):.2f}".rstrip("0").rstrip("."), svg)


def _inject_style(svg: str, css: str) -> str:
    # Mermaid's own <style> is the first one; append ours right after it so
    # ours wins ties without !important gymnastics.
    marker = "</style>"
    index = svg.index(marker) + len(marker)
    return svg[:index] + f"<style>{css}</style>" + svg[index:]


# --- flowcharts -------------------------------------------------------------------


def _node_classes(svg: str) -> dict[str, set[str]]:
    nodes: dict[str, set[str]] = {}
    for match in re.finditer(r'<g class="node ([^"]*)" id="[^"]*-flowchart-([^"]+)-\d+"', svg):
        nodes[match.group(2)] = set(match.group(1).split())
    return nodes


def _edge_target(edge_id: str, nodes: dict[str, set[str]]) -> str | None:
    # ids look like "<prefix>-L_<src>_<dst>_<n>"; node ids may hold "_", so
    # try every split of the middle against the known node ids.
    middle = edge_id.split("-L_", 1)[1].rsplit("_", 1)[0]
    parts = middle.split("_")
    for cut in range(1, len(parts)):
        src, dst = "_".join(parts[:cut]), "_".join(parts[cut:])
        if src in nodes and dst in nodes:
            return dst
    return None


def animate_flowchart(svg: str) -> tuple[str, float]:
    nodes = _node_classes(svg)
    overlays = 0

    def overlay(match: re.Match[str]) -> str:
        tag = match.group(0)
        if "marker-end=" not in tag:  # undirected: a relationship, not a flow
            return tag
        edge_id = _attr(tag, "id") or ""
        target = _edge_target(edge_id, nodes)
        if target is not None and "refuse" in nodes.get(target, set()):
            return tag  # refusals do not carry the flow onward
        nonlocal overlays
        overlays += 1
        thick = "edge-thickness-thick" in _classes(tag)
        d = _attr(tag, "d") or ""
        return f'{tag}<path d="{d}" class="anim-flow{" anim-thick" if thick else ""}"/>'

    svg = re.sub(r'<path\b[^>]*class="[^"]*flowchart-link[^"]*"[^>]*/?>', overlay, svg)
    if overlays == 0:
        raise SystemExit("no directed non-refusal edges found to animate")
    css = (
        f".anim-flow{{fill:none;stroke:{ACCENT};stroke-width:2.5px;stroke-linecap:round;"
        f"stroke-dasharray:6 18;animation:llmr-flow {FLOW_PERIOD_S}s linear infinite;"
        "pointer-events:none}"
        ".anim-flow.anim-thick{stroke-width:4px;stroke-dasharray:8 20;"
        f"animation-name:llmr-flow-thick}}"
        "@keyframes llmr-flow{to{stroke-dashoffset:-24}}"
        "@keyframes llmr-flow-thick{to{stroke-dashoffset:-28}}"
        "@media (prefers-reduced-motion: reduce){.anim-flow{display:none}}"
    )
    return _inject_style(svg, css), FLOW_PERIOD_S


# --- sequence diagrams ------------------------------------------------------------


@dataclass
class _Item:
    span: tuple[int, int]  # the tag's position in the document
    tag: str
    y: float
    draw_length: float | None = None  # a solid <line> to draw, else None


def _line_length(tag: str) -> float:
    x1, y1, x2, y2 = (float(_attr(tag, n) or 0) for n in ("x1", "y1", "x2", "y2"))
    return math.hypot(x2 - x1, y2 - y1)


def _path_start_y(tag: str) -> float:
    match = re.search(r'\bd="M\s*[-\d.]+[ ,]([-\d.]+)', tag)
    return float(match.group(1)) if match else 0.0


def _tag_y(tag: str) -> float:
    if tag.startswith("<line"):
        return float(_attr(tag, "y1") or 0)
    if tag.startswith("<path"):
        return _path_start_y(tag)
    return float(_attr(tag, "y") or 0)


def animate_sequence(svg: str) -> tuple[str, float]:
    """Group the message elements into chronological steps.

    Mermaid emits, per message: its text lines, then the arrow, then the
    sequence-number marker line and text. Notes and activations are emitted
    out of order, so every step is placed by its y coordinate."""
    groups: list[list[_Item]] = []  # one per message, in DOM order
    pending_texts: list[_Item] = []
    expecting_number = 0
    notes: list[list[_Item]] = []
    activations: list[_Item] = []
    for match in _TAG_RE.finditer(svg):
        tag = match.group(0)
        classes = _classes(tag)
        item = _Item(match.span(), tag, _tag_y(tag))
        if "messageText" in classes:
            pending_texts.append(item)
        elif classes & {"messageLine0", "messageLine1"}:
            dashed = "stroke-dasharray" in (_attr(tag, "style") or "")
            if tag.startswith("<line") and not dashed:
                item.draw_length = _line_length(tag)
            groups.append([*pending_texts, item])
            pending_texts = []
            expecting_number = 2
        elif expecting_number and ("sequencenumber" in tag or "sequenceNumber" in classes):
            groups[-1].append(item)
            expecting_number -= 1
        elif "note" in classes and tag.startswith("<rect"):
            notes.append([item])
        elif "noteText" in classes and notes:
            notes[-1].append(item)
        elif "activation0" in classes or "activation1" in classes:
            activations.append(item)
    if not groups:
        raise SystemExit("no sequence messages found to animate")

    # Steps: messages and notes ordered by their y; a message's arrow y is
    # its last (or only) line; a note's is its box top.
    def group_y(items: list[_Item]) -> float:
        arrows = [i for i in items if i.draw_length is not None or i.tag.startswith("<path")]
        return arrows[0].y if arrows else items[0].y

    steps = sorted(
        groups + notes, key=lambda items: group_y(items) if items in groups else items[0].y
    )
    total = len(steps) * SEQ_STEP_S + SEQ_HOLD_S
    starts: dict[int, float] = {id(items): k * SEQ_STEP_S for k, items in enumerate(steps)}
    for activation in activations:
        # Visible from the first message at or below its top edge.
        later = [items for items in groups if group_y(items) >= activation.y - 0.5]
        step = [activation]
        starts[id(step)] = starts[id(later[0])] if later else 0.0
        steps.append(step)

    def pct(seconds: float) -> str:
        return f"{100 * seconds / total:.3f}%"

    css_parts = []
    edits: list[tuple[tuple[int, int], str]] = []
    for k, items in enumerate(steps):
        start = starts[id(items)]
        fade_end = min(start + SEQ_FADE_S, total)
        name = f"llmr-seq-{k}"
        css_parts.append(
            f"@keyframes {name}{{0%,{pct(start)}{{opacity:0}}{pct(fade_end)},100%{{opacity:1}}}}"
        )
        for item in items:
            style = f"animation:{name} {total}s linear infinite;"
            if item.draw_length is not None:
                length = item.draw_length
                draw_end = min(start + SEQ_DRAW_S, total)
                draw = f"{name}-draw"
                css_parts.append(
                    f"@keyframes {draw}{{0%,{pct(start)}{{stroke-dashoffset:{length:.1f}}}"
                    f"{pct(draw_end)},100%{{stroke-dashoffset:0}}}}"
                )
                style = (
                    f"stroke-dasharray:{length:.1f};"
                    f"animation:{name} {total}s linear infinite,{draw} {total}s linear infinite;"
                )
            tag = item.tag
            if 'style="' in tag:
                tag = tag.replace('style="', f'style="{style}', 1)
            else:
                tag = (
                    tag[:-1].rstrip("/")
                    + f' style="{style}"'
                    + ("/>" if tag.endswith("/>") else ">")
                )
            edits.append((item.span, tag))
    for (begin, end), tag in sorted(edits, key=lambda e: e[0][0], reverse=True):
        svg = svg[:begin] + tag + svg[end:]
    css_parts.append("@media (prefers-reduced-motion: reduce){*{animation:none!important}}")
    return _inject_style(svg, "".join(css_parts)), total


# --- rendering ----------------------------------------------------------------------


def _puppeteer_config(tmp: Path) -> Path:
    given = os.environ.get("MMDC_PUPPETEER_CONFIG")
    if given:
        return Path(given)
    config: dict[str, object] = {"args": ["--no-sandbox"]}
    chromium = os.environ.get("PUPPETEER_EXECUTABLE_PATH")
    if chromium:
        config["executablePath"] = chromium
    path = tmp / "puppeteer.json"
    path.write_text(json.dumps(config))
    return path


def render_svg(source: Path, tmp: Path) -> str:
    out = tmp / (source.stem + ".svg")
    subprocess.run(
        [
            "npx",
            "-y",
            MERMAID_CLI,
            "-p",
            str(_puppeteer_config(tmp)),
            "-i",
            str(source),
            "-o",
            str(out),
            "-b",
            "white",
            "--quiet",
        ],
        check=True,
    )
    return out.read_text(encoding="utf-8")


def render_gif(svg: str, period: float, fps: int, out: Path, tmp: Path) -> None:
    from playwright.sync_api import sync_playwright

    box = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"', svg)
    if not box:
        raise SystemExit("SVG without a viewBox")
    width, height = (math.ceil(float(v)) for v in box.groups()[2:])
    frames = tmp / (out.stem + "-frames")
    frames.mkdir(exist_ok=True)
    count = max(2, round(period * fps))
    html = (
        '<!doctype html><html><body style="margin:0;background:white">'
        + re.sub(
            r'<svg\b([^>]*?)\s(?:width|height)="[^"]*"',
            r"<svg\1",
            svg,
            count=2,
        ).replace("<svg", f'<svg width="{width}" height="{height}"', 1)
        + "</body></html>"
    )
    launch: dict[str, object] = {"args": ["--no-sandbox"]}
    chromium = os.environ.get("PUPPETEER_EXECUTABLE_PATH")
    if chromium:
        launch["executable_path"] = chromium
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html)
        for index in range(count):
            at_ms = 1000 * period * index / count
            page.evaluate(
                "t => document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; })",
                at_ms,
            )
            page.screenshot(path=str(frames / f"{index:04d}.png"))
        browser.close()
    _assemble_gif(sorted(frames.glob("*.png")), fps, out)


def _assemble_gif(frame_files: list[Path], fps: int, out: Path) -> None:
    """One shared 128-colour palette (built from the last, most complete
    frame), every frame quantised to it without dither — flat diagram
    colours and text stay crisp — scaled to GIF_MAX_WIDTH."""
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in frame_files]
    width, height = images[0].size
    if width > GIF_MAX_WIDTH:
        size = (GIF_MAX_WIDTH, round(height * GIF_MAX_WIDTH / width))
        images = [image.resize(size, Image.LANCZOS) for image in images]
    palette = images[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantised = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def process(source: Path, tmp: Path, *, gif: bool) -> None:
    svg = _round_numbers(render_svg(source, tmp))
    kind = _attr(re.search(r"<svg\b[^>]*>", svg).group(0), "aria-roledescription") or ""  # type: ignore[union-attr]
    if kind.startswith("flowchart"):
        animated, period = animate_flowchart(svg)
        fps = GIF_FPS_FLOW
    elif kind == "sequence":
        animated, period = animate_sequence(svg)
        fps = GIF_FPS_SEQ
    else:
        raise SystemExit(f"{source}: cannot animate a {kind or 'unknown'} diagram")
    svg_out = source.with_suffix(".svg")
    svg_out.write_text(animated, encoding="utf-8")
    print(f"rendered {svg_out} ({svg_out.stat().st_size // 1024} KiB, {period:.1f}s loop)")
    if gif:
        gif_out = source.with_suffix(".gif")
        render_gif(animated, period, fps, gif_out, tmp)
        print(f"rendered {gif_out} ({gif_out.stat().st_size // 1024} KiB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--diagrams", default="docs/diagrams", help="directory of .mmd sources")
    parser.add_argument("--only", help="render one diagram (its stem)")
    parser.add_argument("--no-gif", action="store_true", help="animated SVG only")
    args = parser.parse_args(argv)
    sources = sorted(Path(args.diagrams).glob("*.mmd"))
    if args.only:
        sources = [s for s in sources if s.stem == args.only]
    chosen = [s for s in sources if ANIMATE_MARKER.search(s.read_text(encoding="utf-8"))]
    if not chosen:
        print("no diagram opts in (add a '%% animate' comment to its .mmd)", file=sys.stderr)
        return 1
    # Next to the diagrams, not the system temp dir: Playwright's driver
    # writes the frames from its own process, and a sandbox may give it a
    # different /tmp.
    with tempfile.TemporaryDirectory(dir=Path(args.diagrams), prefix=".render-") as tmp:
        for source in chosen:
            process(source, Path(tmp), gif=not args.no_gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
