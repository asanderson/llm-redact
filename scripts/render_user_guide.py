"""Render src/llm_redact/user_guide.md -> user_guide.html (package data).

A deliberately CONSTRAINED markdown subset — #/##/### headings,
paragraphs, `- ` lists, fenced code blocks, inline code, **bold**, and
[text](url) links — converted by ~80 lines of stdlib so the runtime needs
no markdown dependency and the output is reviewable. The committed HTML
is pinned against this renderer by tests/test_user_guide.py (the
render_plugins.py discipline): edit the MARKDOWN, re-run this script,
commit both.

The page is self-contained (inline CSS only) so it renders inside the
proxy's strict CSP, and every guide URL stays same-origin.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "llm_redact"

_STYLE = """\
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem auto;
         max-width: 46rem; padding: 0 1rem; line-height: 1.55; color: #1a2330; }
  h1, h2, h3 { line-height: 1.2; } h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem;
         margin-top: 2rem; border-bottom: 1px solid #d8dee7; padding-bottom: .25rem; }
  code { background: #eef1f5; border-radius: 4px; padding: .1rem .3rem; font-size: .9em; }
  pre { background: #eef1f5; border-radius: 6px; padding: .8rem; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  a { color: #0b5fa5; } li { margin: .25rem 0; }
  .top { font-size: .85rem; opacity: .7; margin-bottom: 1.5rem; }
"""

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _inline(text: str) -> str:
    """Escape first, then re-introduce the three inline forms by matching
    on the ESCAPED text — markup can never smuggle raw HTML through."""
    escaped = html.escape(text, quote=False)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    return _LINK_RE.sub(r'<a href="\2">\1</a>', escaped)


def render(markdown: str) -> str:
    out: list[str] = []
    paragraph: list[str] = []
    in_list = False
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in markdown.splitlines():
        if in_code:
            if line.startswith("```"):
                out.append("</code></pre>")
                in_code = False
            else:
                out.append(html.escape(line))
            continue
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            out.append("<pre><code>")
            in_code = True
            continue
        heading = re.match(r"^(#{1,3}) (.*)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue
        if line.startswith("- "):
            flush_paragraph()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if line.startswith("  ") and in_list and line.strip():
            # Continuation of the previous list item.
            out[-1] = out[-1][: -len("</li>")] + f" {_inline(line.strip())}</li>"
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue
        close_list()
        paragraph.append(line.strip())
    flush_paragraph()
    close_list()

    body = "\n".join(out)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>llm-redact user guide</title>\n"
        f"<style>\n{_STYLE}</style>\n</head>\n<body>\n"
        '<p class="top"><a href="/__llm-redact/">&larr; back to the dashboard</a></p>\n'
        f"{body}\n</body>\n</html>\n"
    )


def main() -> None:
    markdown = (PACKAGE / "user_guide.md").read_text(encoding="utf-8")
    target = PACKAGE / "user_guide.html"
    target.write_text(render(markdown), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
