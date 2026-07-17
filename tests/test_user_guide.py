"""The user guide surface: packaged markdown + rendered HTML, the
/__llm-redact/guide endpoint, the CLI command, and the render sync."""

from __future__ import annotations

import importlib.resources
import importlib.util
import sys
from pathlib import Path

import httpx

_SCRIPT = Path(__file__).parents[1] / "scripts" / "render_user_guide.py"
_spec = importlib.util.spec_from_file_location("render_user_guide", _SCRIPT)
assert _spec is not None and _spec.loader is not None
renderer = importlib.util.module_from_spec(_spec)
sys.modules["render_user_guide"] = renderer
_spec.loader.exec_module(renderer)


def _package_text(name: str) -> str:
    return importlib.resources.files("llm_redact").joinpath(name).read_text("utf-8")


def test_committed_html_matches_renderer() -> None:
    """Edit the MARKDOWN and re-run scripts/render_user_guide.py — the
    committed HTML is pinned to the renderer's output (render_plugins
    discipline, both directions by construction)."""
    assert renderer.render(_package_text("user_guide.md")) == _package_text("user_guide.html")


def test_renderer_escapes_raw_html() -> None:
    html = renderer.render("# t\n\n<script>alert(1)</script> and `<code>`\n")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_guide_covers_every_plugin_command() -> None:
    from llm_redact.plugin_assets import COMMANDS

    guide = _package_text("user_guide.md")
    for command in COMMANDS:
        assert f"**{command.name}**" in guide, f"guide is missing the {command.name} command"


async def test_guide_endpoint_serves_html_with_hardening_headers() -> None:
    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    app = create_app(Config())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    response = await client.get("/__llm-redact/guide")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "user guide" in response.text
    await client.aclose()


def test_dashboard_links_to_the_guide() -> None:
    assert 'href="/__llm-redact/guide"' in _package_text("dashboard.html")


def test_guide_ships_as_package_data_and_is_self_contained() -> None:
    # The same importlib.resources lookup the proxy does at startup — fails
    # if either guide file is ever dropped from the package. Self-contained
    # by design: no scripts, no external resources (the only http:// is the
    # loopback dashboard example inside a code span), so it renders under
    # the proxy's strict CSP.
    html = _package_text("user_guide.html")
    assert "<title>llm-redact user guide</title>" in html
    assert "<script" not in html
    assert "src=" not in html
    assert "https://" not in html


def test_cli_guide_prints_the_packaged_markdown(capsys) -> None:
    import argparse

    from llm_redact.cli import run_guide

    assert run_guide(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "# llm-redact user guide" in out
    assert "config editor" in out
