"""The deploy/ ops assets are well-formed and reference only real metrics.

A Grafana panel or an alert rule that names a metric the proxy does not emit
is silently broken — it just shows "No data". This test parses the assets and
asserts every `llm_redact_*` token they reference is an actual metric (allowing
the histogram `_bucket`/`_sum`/`_count` suffixes), so renaming a metric without
updating the dashboard fails CI.
"""

import importlib.util
import inspect
import json
import math
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from llm_redact import __version__, proxy
from llm_redact.config import Config
from llm_redact.proxy import create_app

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
HELM_CHART = DEPLOY / "helm" / "llm-redact"
_HIST_SUFFIXES = ("_bucket", "_sum", "_count")


async def _emitted_metric_names() -> set[str]:
    app = create_app(Config())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p")
    text = (await client.get("/__llm-redact/metrics")).text
    await client.aclose()
    return {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE ")}


def _referenced_metrics(text: str) -> set[str]:
    return set(re.findall(r"llm_redact_[a-z_]+", text))


def _canonical(name: str) -> str:
    for suffix in _HIST_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def test_grafana_dashboard_is_valid_json() -> None:
    dashboard = json.loads((DEPLOY / "grafana-dashboard.json").read_text())
    assert dashboard["uid"] == "llm-redact"
    assert dashboard["panels"], "dashboard has no panels"


def _dashboard_panels() -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = json.loads((DEPLOY / "grafana-dashboard.json").read_text())[
        "panels"
    ]
    return panels


def test_dashboard_panels_fit_the_grid_without_overlap() -> None:
    # Grafana does not refuse an import whose panels overlap: it shoves one
    # below the other, so a panel added at a taken spot lands somewhere else
    # (or under the wrong row) without a word. Every panel fits the 24-column
    # grid, none covers another, an id (Grafana assigns one on import when a
    # panel has none) is never shared, and each queries the data source the
    # importer picks.
    panels = _dashboard_panels()
    boxes = []
    for panel in panels:
        grid = panel["gridPos"]
        x, y, w, h = grid["x"], grid["y"], grid["w"], grid["h"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 24, panel["title"]
        boxes.append((panel["title"], x, y, w, h))
    for index, (title, x, y, w, h) in enumerate(boxes):
        for other, ox, oy, ow, oh in boxes[index + 1 :]:
            apart = x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
            assert apart, f"{title!r} overlaps {other!r}"
    ids = [panel["id"] for panel in panels if "id" in panel]
    assert len(ids) == len(set(ids)), ids
    assert len({panel["title"] for panel in panels}) == len(panels)
    for panel in panels:
        if panel["type"] != "row":
            assert panel["datasource"] == {"type": "prometheus", "uid": "${datasource}"}, panel


@pytest.mark.anyio
async def test_deploy_assets_reference_only_real_metrics() -> None:
    emitted = await _emitted_metric_names()
    for asset in ("grafana-dashboard.json", "prometheus-alerts.yml"):
        referenced = _referenced_metrics((DEPLOY / asset).read_text())
        assert referenced, f"{asset} references no llm_redact metrics"
        unknown = {m for m in referenced if _canonical(m) not in emitted}
        assert not unknown, f"{asset} references metrics the proxy does not emit: {sorted(unknown)}"


def test_no_alert_reads_the_end_to_end_duration() -> None:
    # llm_redact_request_duration_seconds includes the provider's own time
    # (a model thinking, a stream's length): an alert on it fires on
    # ordinary LLM traffic. Latency is alerted on the proxy's own share.
    rules = yaml.safe_load((DEPLOY / "prometheus-alerts.yml").read_text())["groups"][0]["rules"]
    by_name = {rule["alert"]: rule for rule in rules}
    assert not [r for r in rules if "llm_redact_request_duration_seconds" in r["expr"]]
    overhead = by_name["LlmRedactHighProxyOverheadP95"]
    assert "llm_redact_proxy_overhead_seconds_bucket" in overhead["expr"]
    assert "LlmRedactHighLatencyP95" not in by_name
    storage = by_name["LlmRedactLocalFaultRefusals"]
    assert storage["labels"]["severity"] == "critical"
    assert 'kind=~"audit_unavailable|vault_fault"' in storage["expr"]


def test_ner_coverage_gap_has_panels_and_an_alert() -> None:
    # Strings longer than [detection.ner] max_chars are never read by the
    # models: the shipped dashboard charts every NER family (below) and an
    # alert pages on a sustained skip rate, with a runbook-style description.
    rules = _alert_rules()
    rule = rules["LlmRedactNerSkippingLongStrings"]
    assert 'llm_redact_ner_strings_total{outcome="skipped_max_chars"}' in rule["expr"]
    assert rule["labels"]["severity"] == "warning"
    assert rule["annotations"]["summary"]
    assert "max_chars" in rule["annotations"]["description"]
    # The llm backend's model server failing: refused or forwarded without
    # its findings, either way worth a look.
    server = rules["LlmRedactNerServerFailing"]
    assert "llm_redact_ner_errors_total" in server["expr"]
    assert server["labels"]["severity"] == "warning"
    assert "on_error" in server["annotations"]["description"]


def _alert_rules() -> dict[str, dict[str, Any]]:
    rules = yaml.safe_load((DEPLOY / "prometheus-alerts.yml").read_text())["groups"][0]["rules"]
    return {rule["alert"]: rule for rule in rules}


def _row_panels(title: str) -> list[dict[str, Any]]:
    """The panels laid out below the row titled ``title``, up to the next row."""
    panels = _dashboard_panels()
    rows = [p["gridPos"]["y"] for p in panels if p["type"] == "row"]
    top = next(p["gridPos"]["y"] for p in panels if p["type"] == "row" and p["title"] == title)
    bottom = min((y for y in rows if y > top), default=math.inf)
    return [p for p in panels if p["type"] != "row" and top < p["gridPos"]["y"] < bottom]


def _charted_series(expr: str) -> str:
    """What a query or an alert's expression charts: its comparison
    (``> 0``) and the length of its rate windows left out, so an alert
    over 15 minutes and a panel over 5 read the same series."""
    series = re.sub(r"\s*>\s*0\s*$", "", expr.strip())
    return re.sub(r"\[\d+[smh]\]", "[window]", series)


# Field overrides that change how a series is drawn or whether it is drawn
# at all: its color, and Grafana's "Hide in area" (legend, tooltip, graph).
_SERIES_LOOKS = frozenset({"color", "custom.hideFrom"})


def _drawn_in_the_panel_color(panel: dict[str, Any], target: dict[str, Any]) -> None:
    """Assert ``target``'s series is drawn, in ``panel``'s default color.

    An override may recolor or hide only OTHER queries' series, named by
    their refId (``byFrameRefID``): one matching fields by name, by a
    regular expression, by type or by value (what Grafana's editor offers
    first, and exports as is) can reach this series whatever its refId, and
    so can one naming this target's own refId. Neither a transformation (it
    can drop or rename the series) nor a hidden query (Grafana never runs
    it) is allowed on the panel either."""
    where = (panel["title"], target["refId"])
    assert not target.get("hide"), where
    assert not panel.get("transformations"), where
    assert not panel["fieldConfig"]["defaults"].get("custom", {}).get("hideFrom"), where
    for override in panel["fieldConfig"]["overrides"]:
        if not {prop["id"] for prop in override["properties"]} & _SERIES_LOOKS:
            continue
        matcher = override["matcher"]
        assert matcher["id"] == "byFrameRefID", (where, matcher)
        assert matcher["options"] != target["refId"], (where, matcher)


_NER_ALERTS = ("LlmRedactNerSkippingLongStrings", "LlmRedactNerServerFailing")
_ORANGE = {"mode": "fixed", "fixedColor": "orange"}


def test_the_ner_coverage_row_charts_every_ner_family() -> None:
    # Every llm_redact_ner_* family the core renders (CORE_METRIC_FAMILIES
    # is pinned to the rendered set by test_metrics_surfaces.py) is charted
    # in the NER coverage row, and nowhere else: a family a new release
    # adds fails here until it is on the dashboard.
    from llm_redact.metrics import CORE_METRIC_FAMILIES

    families = {name for name in CORE_METRIC_FAMILIES if name.startswith("llm_redact_ner_")}
    row = _row_panels("NER coverage")
    charted = [m for p in row for t in p["targets"] for m in _referenced_metrics(t["expr"])]
    assert set(charted) == families, set(charted) ^ families
    elsewhere = [
        p["title"]
        for p in _dashboard_panels()
        if p not in row and "llm_redact_ner_" in json.dumps(p.get("targets", []))
    ]
    assert not elsewhere, elsewhere
    for panel in row:
        for target in panel["targets"]:
            assert "sum by (backend" in target["expr"], target
            assert target["legendFormat"].startswith("{{backend}}"), target


def test_the_ner_rules_series_are_charted_in_the_warning_color() -> None:
    # The series each NER alert reads is charted by exactly one target of
    # the dashboard, on an NER coverage panel that names the alert and
    # draws it in the dashboard's warning color (orange): never recolored
    # or hidden by an override, a transformation or a hidden query.
    row = _row_panels("NER coverage")
    rules = _alert_rules()
    for alert in _NER_ALERTS:
        expr = rules[alert]["expr"]
        assert re.search(r">\s*0\s*$", expr), alert
        wanted = _charted_series(expr)
        found = [
            (p, t)
            for p in _dashboard_panels()
            for t in p.get("targets", [])
            if _charted_series(t["expr"]) == wanted
        ]
        assert len(found) == 1, (alert, [p["title"] for p, _ in found])
        panel, target = found[0]
        assert panel in row, alert
        assert alert in panel["description"], alert
        assert panel["fieldConfig"]["defaults"]["color"] == _ORANGE, alert
        _drawn_in_the_panel_color(panel, target)


def test_the_ner_model_server_has_a_panel_of_its_own() -> None:
    # The llm backend's model server failures are charted apart from the
    # entities a model returned and the redaction dropped (one panel held
    # all six series once, the alert's among them): server errors in
    # orange (LlmRedactNerServerFailing), the breaker's short-circuited
    # strings yellow and its openings red, each of the two recolored by its
    # own refId only — so no override reaches the alert's series.
    row = _row_panels("NER coverage")
    server = [p for p in row if "llm_redact_ner_errors_total" in json.dumps(p["targets"])]
    (panel,) = server
    exprs = {t["refId"]: t["expr"] for t in panel["targets"]}
    assert sorted(m for e in exprs.values() for m in _referenced_metrics(e)) == [
        "llm_redact_ner_breaker_opens_total",
        "llm_redact_ner_breaker_short_circuits_total",
        "llm_redact_ner_errors_total",
    ]
    errors = next(ref for ref, e in exprs.items() if "llm_redact_ner_errors_total" in e)
    colors: dict[str, str] = {}
    for override in panel["fieldConfig"]["overrides"]:
        assert override["matcher"]["id"] == "byFrameRefID", override
        (prop,) = override["properties"]
        assert prop["id"] == "color", override
        colors[override["matcher"]["options"]] = prop["value"]["fixedColor"]
    breaker = {
        "llm_redact_ner_breaker_short_circuits_total": "yellow",
        "llm_redact_ner_breaker_opens_total": "red",
    }
    assert errors not in colors
    assert {_referenced_metrics(exprs[ref]).pop(): color for ref, color in colors.items()} == (
        breaker
    )
    assert "[detection.ner.llm]" in panel["description"]
    for family in (*breaker, "llm_redact_ner_errors_total"):
        assert family in panel["description"], family
    # The entities panel no longer holds a model-server series.
    (dropped,) = [p for p in row if "llm_redact_ner_labels_dropped_total" in json.dumps(p)]
    assert not {m for t in dropped["targets"] for m in _referenced_metrics(t["expr"])} & {
        *breaker,
        "llm_redact_ner_errors_total",
    }


def test_the_entities_dropped_panel_says_what_ungrounded_counts() -> None:
    # labels and offsets dropped mean a configuration or a model is wrong:
    # they should stay at zero. ungrounded also counts what the llm
    # backend's grounding rightly refuses (llm_ner._grounded): a value
    # shorter than two characters, or one holding a placeholder's guillemet
    # (an earlier turn's token a request carries). So the panel names every
    # case the metrics reference names, and its "should stay at zero" leaves
    # ungrounded out (one sentence once held all three, naming neither case).
    row = _row_panels("NER coverage")
    (panel,) = [p for p in row if "llm_redact_ner_ungrounded_total" in json.dumps(p["targets"])]
    description = panel["description"]
    (zero,) = [s for s in re.split(r"(?<=\.)\s+", description) if "stay at zero" in s]
    assert "labels" in zero and "offsets" in zero, zero
    assert "ungrounded" not in zero, zero
    doc = (DEPLOY.parent / "docs" / "observability.md").read_text(encoding="utf-8")
    (reference,) = [
        line
        for line in doc.splitlines()
        if line.startswith("| `llm_redact_ner_ungrounded_total` |")
    ]
    for phrase in ("two characters", "guillemet"):
        assert phrase in reference, phrase
        assert phrase in description, phrase


def _override(matcher: str, options: Any, prop: str, value: Any) -> dict[str, Any]:
    return {
        "matcher": {"id": matcher, "options": options},
        "properties": [{"id": prop, "value": value}],
    }


_GREEN = {"mode": "fixed", "fixedColor": "green"}
_HIDDEN = {"legend": True, "tooltip": True, "viz": True}


@pytest.mark.parametrize(
    ("change", "drawn"),
    [
        pytest.param(lambda panel: None, True, id="as-shipped"),
        # What Grafana's override editor offers first, exported as is: the
        # series matched by its display name or a pattern, whatever its refId.
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byRegexp", ".*server errors", "color", _GREEN)
            ),
            False,
            id="recolored-by-pattern",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byName", "llm server errors", "color", _GREEN)
            ),
            False,
            id="recolored-by-name",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byRegexp", ".*", "custom.hideFrom", _HIDDEN)
            ),
            False,
            id="hidden-by-pattern",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["defaults"].update(custom={"hideFrom": _HIDDEN}),
            False,
            id="hidden-by-default",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byValue", {"reducer": "max", "op": "gte", "value": 0}, "color", _GREEN)
            ),
            False,
            id="recolored-by-value",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byFrameRefID", "A", "color", _GREEN)
            ),
            False,
            id="recolored-by-its-refid",
        ),
        pytest.param(lambda panel: panel["targets"][0].update(hide=True), False, id="query-hidden"),
        pytest.param(
            lambda panel: panel.update(
                transformations=[{"id": "filterFieldsByName", "options": {}}]
            ),
            False,
            id="transformed",
        ),
        # What leaves the alert's series alone: another query recolored by
        # its refId, a unit by name.
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byFrameRefID", "B", "color", _GREEN)
            ),
            True,
            id="another-query-recolored",
        ),
        pytest.param(
            lambda panel: panel["fieldConfig"]["overrides"].append(
                _override("byName", "llm server errors", "unit", "short")
            ),
            True,
            id="unit-by-name",
        ),
    ],
)
def test_no_override_can_recolor_or_hide_the_series_an_ner_alert_reads(
    change: Callable[[dict[str, Any]], None], drawn: bool
) -> None:
    # The NER model server panel as shipped, its errors query (refId A:
    # what LlmRedactNerServerFailing reads) edited the ways a Grafana export
    # can: the check above must notice every one that reaches the alert's
    # series, and only those.
    panel = next(
        p
        for p in _row_panels("NER coverage")
        if "llm_redact_ner_errors_total" in json.dumps(p["targets"])
    )
    target = panel["targets"][0]
    assert target["refId"] == "A" and "llm_redact_ner_errors_total" in target["expr"]
    change(panel)
    if drawn:
        _drawn_in_the_panel_color(panel, target)
    else:
        with pytest.raises(AssertionError):
            _drawn_in_the_panel_color(panel, target)


def test_the_ner_prefetch_counters_are_charted() -> None:
    # Where the NER models ran — on a request's path, inline is the event
    # loop, holding up every other request — is charted beside the strings
    # they read, both families on one panel whose description says what
    # each counts (docs/how-it-works.md, the NER prefetch) and that the
    # pro dashboard's preview runs inline too (off the loop, but counted).
    row = _row_panels("NER coverage")
    families = ("llm_redact_ner_inline_calls_total", "llm_redact_ner_prefetch_misses_total")
    (panel,) = [p for p in row if any(f in json.dumps(p["targets"]) for f in families)]
    assert sorted(m for t in panel["targets"] for m in _referenced_metrics(t["expr"])) == list(
        families
    )
    for target in panel["targets"]:
        (family,) = _referenced_metrics(target["expr"])
        assert target["expr"] == f"sum by (backend) (rate({family}[5m]))", target
    strings = next(p for p in row if p["title"] == "NER strings by backend and outcome")
    assert panel["gridPos"]["y"] == strings["gridPos"]["y"]
    description = panel["description"]
    for phrase in (*families, "event loop", "prefetch", "preview", "reload"):
        assert phrase in description, phrase
    assert "docs/how-it-works.md" in description
    assert "LlmRedact" not in description  # no alert reads them: not a warning panel
    assert panel["fieldConfig"]["defaults"].get("color") is None


def test_dropped_scan_captures_have_an_alert() -> None:
    # A plugin's scan sinks that cannot keep up lose captures: the requests
    # are unaffected, so the alert says so and names the queue's metrics;
    # a process's last drain (reason shutdown) is no trend.
    rules = yaml.safe_load((DEPLOY / "prometheus-alerts.yml").read_text())["groups"][0]["rules"]
    rule = {r["alert"]: r for r in rules}["LlmRedactScanCapturesDropped"]
    assert 'llm_redact_scan_captures_dropped_total{reason!="shutdown"}' in rule["expr"]
    assert rule["labels"]["severity"] == "warning"
    assert rule["annotations"]["summary"]
    description = rule["annotations"]["description"]
    assert "requests are unaffected" in description
    assert "llm_redact_scan_capture_queue_depth" in description
    assert 'llm_redact_bookkeeping_errors_total{stage="scan_sink"}' in description


def test_scan_capture_panels_read_what_the_alert_reads() -> None:
    # The dashboard's Scan sinks row charts all three families. The drops
    # panel colors the reasons LlmRedactScanCapturesDropped pages on (every
    # reason but shutdown) like the dashboard's other warning panels, and
    # the shutdown reason (a process's last drain, never a trend) apart.
    # The families have samples only while a plugin registers a sink, so
    # each panel says why it is empty in the Free core.
    panels = _dashboard_panels()
    row = next(p for p in panels if p["type"] == "row" and "Scan sinks" in p["title"])
    by_family = {
        family: [p for p in panels if any(family in t["expr"] for t in p.get("targets", []))]
        for family in (
            "llm_redact_scan_captures_delivered_total",
            "llm_redact_scan_captures_dropped_total",
            "llm_redact_scan_capture_queue_depth",
        )
    }
    for family, found in by_family.items():
        assert len(found) == 1, family
        panel = found[0]
        assert panel["gridPos"]["y"] == row["gridPos"]["y"] + row["gridPos"]["h"], family
        assert family in panel["description"]
        assert "only while a plugin registers a scan sink" in panel["description"]
    delivered = by_family["llm_redact_scan_captures_delivered_total"][0]
    assert "rate(llm_redact_scan_captures_delivered_total[" in delivered["targets"][0]["expr"]
    assert "UNREDACTED" in delivered["description"]
    depth = by_family["llm_redact_scan_capture_queue_depth"][0]
    assert depth["fieldConfig"]["defaults"]["unit"] == "short"
    assert "rate(" not in depth["targets"][0]["expr"]

    dropped = by_family["llm_redact_scan_captures_dropped_total"][0]
    rules = yaml.safe_load((DEPLOY / "prometheus-alerts.yml").read_text())["groups"][0]["rules"]
    alert = {r["alert"]: r for r in rules}["LlmRedactScanCapturesDropped"]
    selector = 'llm_redact_scan_captures_dropped_total{reason!="shutdown"}'
    assert selector in alert["expr"]
    targets = {t["refId"]: t for t in dropped["targets"]}
    alerting = [ref for ref, t in targets.items() if selector in t["expr"]]
    assert len(alerting) == 1
    _drawn_in_the_panel_color(dropped, targets[alerting[0]])
    (shutdown,) = (ref for ref in targets if ref not in alerting)
    assert 'reason="shutdown"' in targets[shutdown]["expr"]
    for target in targets.values():
        assert "sum by (reason) (rate(" in target["expr"]
        assert target["legendFormat"] == "{{reason}}"
    defaults = dropped["fieldConfig"]["defaults"]
    assert defaults["color"] == {"mode": "fixed", "fixedColor": "orange"}
    (override,) = dropped["fieldConfig"]["overrides"]
    assert override["matcher"] == {"id": "byFrameRefID", "options": shutdown}
    (color,) = override["properties"]
    assert color["id"] == "color" and color["value"]["fixedColor"] != "orange"
    assert "LlmRedactScanCapturesDropped" in dropped["description"]
    assert "Requests are unaffected" in dropped["description"]


def test_prometheus_scrape_targets_reserved_path() -> None:
    text = (DEPLOY / "prometheus-scrape.yml").read_text()
    assert "/__llm-redact/metrics" in text
    assert "job_name: llm-redact" in text


def test_k8s_sidecar_hardening_strings_present() -> None:
    # Stdlib-only guard (always runs in CI): the load-bearing hardening
    # directives and the real health-probe paths must be in the manifest.
    text = (DEPLOY / "k8s-sidecar.yaml").read_text()
    for needle in (
        "readOnlyRootFilesystem: true",
        "allowPrivilegeEscalation: false",
        'drop: ["ALL"]',
        "runAsNonRoot: true",
        "/__llm-redact/healthz",
        "/__llm-redact/readyz",
    ):
        assert needle in text, f"k8s manifest missing hardening directive: {needle!r}"


def test_pyyaml_is_a_direct_dev_dependency() -> None:
    # The structural k8s/Helm tests parse YAML. PyYAML once arrived only
    # through libcst (via mutmut), which on Python 3.13 — the CI helm job's —
    # pulls pyyaml-ft (module `yaml_ft`) instead, so every render test there
    # skipped on importorskip and the chart was never checked rendered.
    pyproject = tomllib.loads((DEPLOY.parent / "pyproject.toml").read_text())
    dev = pyproject["dependency-groups"]["dev"]
    assert any(re.match(r"pyyaml\b", dep, re.IGNORECASE) for dep in dev)
    lock = (DEPLOY.parent / "uv.lock").read_text()
    assert '{ name = "pyyaml", specifier = ">=6" }' in lock


def test_k8s_sidecar_is_hardened_and_probes_real_endpoints() -> None:
    doc = next(yaml.safe_load_all((DEPLOY / "k8s-sidecar.yaml").read_text()))
    assert doc["kind"] == "Deployment"
    spec = doc["spec"]["template"]["spec"]
    redact = next(c for c in spec["containers"] if c["name"] == "llm-redact")
    # Container hardening the manifest promises.
    sec = redact["securityContext"]
    assert sec["readOnlyRootFilesystem"] is True
    assert sec["allowPrivilegeEscalation"] is False
    assert sec["capabilities"]["drop"] == ["ALL"]
    assert spec["securityContext"]["runAsNonRoot"] is True
    # 3.3.0 bind honesty: the sidecar binds LOOPBACK (other pods cannot reach
    # it), needs no INSECURE_BIND hatch, and therefore uses exec probes —
    # kubelet httpGet dials the pod IP, which 127.0.0.1 does not answer.
    env = {e["name"]: e.get("value") for e in redact["env"]}
    assert env["LLM_REDACT_HOST"] == "127.0.0.1"
    assert "LLM_REDACT_INSECURE_BIND" not in env
    assert "/__llm-redact/healthz" in redact["livenessProbe"]["exec"]["command"][-1]
    assert "/__llm-redact/readyz" in redact["readinessProbe"]["exec"]["command"][-1]


def test_k8s_sidecar_pins_the_image_of_this_release() -> None:
    # The manifest names the exact image it deploys, and the release-prep
    # commit moves that tag with __version__ (docs/RELEASING.md). It once named
    # a tag that was never published, so a pod applied from it could not pull.
    doc = next(yaml.safe_load_all((DEPLOY / "k8s-sidecar.yaml").read_text()))
    spec = doc["spec"]["template"]["spec"]
    redact = next(c for c in spec["containers"] if c["name"] == "llm-redact")
    assert redact["image"] == f"ghcr.io/asanderson/llm-redact:{__version__}"


# --- Helm chart (deploy/helm/llm-redact) ------------------------------------
#
# The needle + appVersion tests are stdlib-only and always run. The rendering
# tests shell out to `helm` and skip when it is absent (the CI `helm` job runs
# them for real); they parse the output with PyYAML, a direct dev dependency
# (never importorskip'd: a missing module must fail, not skip unseen).

_HELM = shutil.which("helm")
_needs_helm = pytest.mark.skipif(_HELM is None, reason="helm not installed")


def _chart_field(field: str) -> str:
    # Stdlib scalar extraction from Chart.yaml — no yaml dep for the pin that
    # keeps the chart tag in lockstep with the package version.
    text = (HELM_CHART / "Chart.yaml").read_text()
    match = re.search(rf'^{field}:\s*"?([^"\n]+)"?\s*$', text, re.MULTILINE)
    assert match, f"Chart.yaml missing {field}"
    return match.group(1).strip()


def _helm_template(*set_args: str) -> subprocess.CompletedProcess[str]:
    cmd = ["helm", "template", "rel", str(HELM_CHART)]
    for kv in set_args:
        cmd += ["--set", kv]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_helm_chart_appversion_tracks_package_version() -> None:
    # The default image tag is .Chart.AppVersion; a version bump must carry the
    # chart with it or `helm install` pulls a stale image (the plugin-json pin).
    assert _chart_field("appVersion") == __version__
    assert _chart_field("version") == __version__


def test_helm_chart_hardening_and_guardrail_present() -> None:
    # Stdlib needle over the templates — always runs, helm not required. The
    # shared hardened container spec and the never-wrong-value guardrail are
    # the load-bearing pieces.
    helpers = (HELM_CHART / "templates" / "_helpers.tpl").read_text()
    for needle in (
        "readOnlyRootFilesystem: true",
        "allowPrivilegeEscalation: false",
        'drop: ["ALL"]',
        "/__llm-redact/healthz",
        "/__llm-redact/readyz",
    ):
        assert needle in helpers, f"chart _helpers.tpl missing: {needle!r}"
    # Pod-level hardening ships as the values.yaml default.
    assert "runAsNonRoot: true" in (HELM_CHART / "values.yaml").read_text()
    # The guardrail must {{ fail }} a per-pod-vault multi-replica render.
    assert "fail " in helpers
    assert "never-wrong-value" in helpers


def test_helm_chart_claims_no_license_gate() -> None:
    # The FOSS core is ungated on Kubernetes and beyond loopback
    # (test_no_gates.py): the chart's NOTES and values must not claim a tier
    # requirement the proxy does not enforce (NOTES once said k8s needed Team).
    notes = (HELM_CHART / "templates" / "NOTES.txt").read_text()
    values = (HELM_CHART / "values.yaml").read_text()
    for text in (notes, values):
        flat = " ".join(text.split())
        assert not re.search(r"\bTeam\b", flat)
        assert "refuses to start on k8s" not in flat
        assert "bind (Pro)" not in flat
    assert "The FOSS core needs no license" in " ".join(notes.split())


def test_helm_standalone_lists_its_service_names_as_allowed_hosts() -> None:
    # Stdlib needle: a standalone proxy is dialled by its Service's DNS
    # names, and a request spending a credential the proxy holds (identity
    # auth via IRSA / Workload Identity) is refused under any name the proxy
    # does not answer to — so the generated config must list them.
    helpers = (HELM_CHART / "templates" / "_helpers.tpl").read_text()
    assert "allowed_hosts = {{ toJson $hosts }}" in helpers
    assert ".svc.cluster.local" in helpers
    assert "allowedHosts: []" in (HELM_CHART / "values.yaml").read_text()


def test_helm_hpa_targets_the_deployment() -> None:
    hpa = (HELM_CHART / "templates" / "hpa.yaml").read_text()
    assert "kind: HorizontalPodAutoscaler" in hpa
    assert "scaleTargetRef" in hpa
    assert "llm-redact.fullname" in hpa  # scaleTargetRef name == the Deployment


@_needs_helm
def test_helm_lint_passes() -> None:
    result = subprocess.run(["helm", "lint", str(HELM_CHART)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@_needs_helm
def test_helm_sidecar_preset_renders() -> None:
    result = _helm_template()
    assert result.returncode == 0, result.stderr
    docs = {d["kind"]: d for d in yaml.safe_load_all(result.stdout) if d}
    assert "Deployment" in docs
    # Sidecar is loopback-only: no Service, no HPA.
    assert "Service" not in docs
    assert "HorizontalPodAutoscaler" not in docs
    # 3.3.0 bind honesty: the sidecar BINDS 127.0.0.1 (the "never exposed"
    # claim is structural, not aspirational), needs no INSECURE_BIND, and
    # uses exec probes because kubelet httpGet dials the pod IP.
    proxy = next(
        c
        for c in docs["Deployment"]["spec"]["template"]["spec"]["containers"]
        if c["name"] == "llm-redact"
    )
    env = {e["name"]: e.get("value") for e in proxy["env"]}
    assert env["LLM_REDACT_HOST"] == "127.0.0.1"
    assert "LLM_REDACT_INSECURE_BIND" not in env
    assert "exec" in proxy["livenessProbe"]
    assert "exec" in proxy["readinessProbe"]


@_needs_helm
def test_helm_standalone_binds_wide_with_httpget_probes() -> None:
    result = _helm_template("mode=standalone", "vault.backend=postgresql")
    assert result.returncode == 0, result.stderr
    docs = {d["kind"]: d for d in yaml.safe_load_all(result.stdout) if d}
    proxy = next(
        c
        for c in docs["Deployment"]["spec"]["template"]["spec"]["containers"]
        if c["name"] == "llm-redact"
    )
    env = {e["name"]: e.get("value") for e in proxy["env"]}
    assert env["LLM_REDACT_HOST"] == "0.0.0.0"  # cross-pod reach is the point
    assert env["LLM_REDACT_INSECURE_BIND"] == "1"  # default hatch (documented)
    assert proxy["livenessProbe"]["httpGet"]["path"] == "/__llm-redact/healthz"


@_needs_helm
def test_helm_optional_hardening_templates_render() -> None:
    result = _helm_template(
        "mode=standalone",
        "vault.backend=postgresql",
        "networkPolicy.enabled=true",
        "podDisruptionBudget.enabled=true",
        "serviceAccount.create=true",
    )
    assert result.returncode == 0, result.stderr
    kinds = {d["kind"] for d in yaml.safe_load_all(result.stdout) if d}
    assert {"NetworkPolicy", "PodDisruptionBudget", "ServiceAccount"} <= kinds


@_needs_helm
def test_helm_extra_volumes_wire_through() -> None:
    # extraVolumes/extraVolumeMounts exist so the chart's own "prefer mTLS"
    # advice is actually wireable — a [tls] cert Secret must be mountable.
    result = _helm_template(
        "extraVolumes[0].name=tls",
        "extraVolumes[0].secret.secretName=llm-redact-tls",
        "extraVolumeMounts[0].name=tls",
        "extraVolumeMounts[0].mountPath=/etc/llm-redact/tls",
    )
    assert result.returncode == 0, result.stderr
    docs = {d["kind"]: d for d in yaml.safe_load_all(result.stdout) if d}
    spec = docs["Deployment"]["spec"]["template"]["spec"]
    proxy = next(c for c in spec["containers"] if c["name"] == "llm-redact")
    assert any(m["name"] == "tls" for m in proxy["volumeMounts"])
    assert any(v["name"] == "tls" for v in spec["volumes"])


@_needs_helm
def test_helm_standalone_autoscaling_preset_renders() -> None:
    result = _helm_template(
        "mode=standalone",
        "autoscaling.enabled=true",
        "vault.backend=postgresql",
        "serviceMonitor.enabled=true",
    )
    assert result.returncode == 0, result.stderr
    docs = {d["kind"]: d for d in yaml.safe_load_all(result.stdout) if d}
    assert {"Deployment", "Service", "HorizontalPodAutoscaler", "ServiceMonitor"} <= set(docs)
    hpa = docs["HorizontalPodAutoscaler"]["spec"]
    assert hpa["scaleTargetRef"]["kind"] == "Deployment"
    assert hpa["scaleTargetRef"]["name"] == docs["Deployment"]["metadata"]["name"]
    # The HPA owns replicas — the Deployment must not pin them.
    assert "replicas" not in docs["Deployment"]["spec"]
    # ServiceMonitor scrapes the real metrics endpoint.
    assert docs["ServiceMonitor"]["spec"]["endpoints"][0]["path"] == "/__llm-redact/metrics"
    # The rendered proxy container keeps every hardening directive.
    proxy = next(
        c
        for c in docs["Deployment"]["spec"]["template"]["spec"]["containers"]
        if c["name"] == "llm-redact"
    )
    sec = proxy["securityContext"]
    assert sec["readOnlyRootFilesystem"] is True
    assert sec["allowPrivilegeEscalation"] is False
    assert sec["capabilities"]["drop"] == ["ALL"]


@_needs_helm
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_helm_standalone_autoscaling_rejects_perpod_vault(backend: str) -> None:
    # never-wrong-value at deploy time: an autoscaled standalone proxy on a
    # per-pod vault would issue divergent tokens per replica — the render fails.
    result = _helm_template(
        "mode=standalone", "autoscaling.enabled=true", f"vault.backend={backend}"
    )
    assert result.returncode != 0
    assert "SHARED vault" in result.stderr


@_needs_helm
def test_helm_standalone_multireplica_rejects_perpod_vault() -> None:
    result = _helm_template("mode=standalone", "replicaCount=3", "vault.backend=sqlite")
    assert result.returncode != 0
    assert "SHARED vault" in result.stderr


@_needs_helm
def test_helm_standalone_single_replica_sqlite_allowed() -> None:
    # A single standalone replica on sqlite is fine — no cross-pod divergence.
    result = _helm_template("mode=standalone", "vault.backend=sqlite")
    assert result.returncode == 0, result.stderr


def _rendered_config(*set_args: str) -> Config:
    """The proxy config the chart renders, parsed by the production parser."""
    import tomllib

    from llm_redact.config import parse_config

    cmd = ["helm", "template", "rel", str(HELM_CHART), "--namespace", "team"]
    for kv in set_args:
        cmd += ["--set", kv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    [configmap] = [d for d in yaml.safe_load_all(result.stdout) if d and d["kind"] == "ConfigMap"]
    return parse_config(tomllib.loads(configmap["data"]["config.toml"]), "<chart>")


@_needs_helm
def test_helm_allowed_hosts_render_and_parse() -> None:
    standalone = _rendered_config("mode=standalone", "vault.backend=postgresql")
    assert standalone.allowed_hosts == (
        "rel-llm-redact",
        "rel-llm-redact.team",
        "rel-llm-redact.team.svc",
        "rel-llm-redact.team.svc.cluster.local",
    )
    # Sidecar: the tool dials 127.0.0.1, nothing to list unless asked.
    assert _rendered_config().allowed_hosts == ()
    extra = _rendered_config("allowedHosts[0]=Redact.Example", "mode=sidecar")
    assert extra.allowed_hosts == ("redact.example",)


# --- Helm termination grace period (the proxy's shutdown budget) ------------


def _shutdown_budget_seconds() -> int:
    """The most the proxy's BOUNDED shutdown steps take once its in-flight
    requests have finished (docs/resilience.md, "Shutdown order"), from the
    constants themselves: the vault's map drain, the sinks' final-flush
    deadline and cancel grace, then the vault close's own drain (it drains
    again when a write landed since the first one) and its thread join."""
    from llm_redact.proxy import _SINK_CANCEL_GRACE_SECONDS, _SINK_CLOSE_TIMEOUT_SECONDS
    from llm_redact.vault_writer import SHUTDOWN_DRAIN_SECONDS, STOP_JOIN_SECONDS

    return math.ceil(
        SHUTDOWN_DRAIN_SECONDS
        + _SINK_CLOSE_TIMEOUT_SECONDS
        + _SINK_CANCEL_GRACE_SECONDS
        + SHUTDOWN_DRAIN_SECONDS
        + STOP_JOIN_SECONDS
    )


def _default_grace_seconds() -> int:
    text = (HELM_CHART / "values.yaml").read_text()
    (value,) = re.findall(r"^terminationGracePeriodSeconds:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(value)


def test_helm_grace_period_default_covers_the_shutdown_budget() -> None:
    # Kubernetes' own default (30 s) SIGKILLs a pod whose audit sink is slow
    # before the audit database and the vault close. The chart's default must
    # cover the bounded steps AND leave room for in-flight requests (the
    # drain before them has no bound of its own) — raising the 45 s sink
    # deadline fails here until the chart follows.
    budget = _shutdown_budget_seconds()
    assert budget == 57  # the number values.yaml, NOTES and the docs quote
    assert _default_grace_seconds() > budget
    assert _default_grace_seconds() == 90


def test_helm_grace_period_is_wired_and_validated() -> None:
    # Stdlib needles: the pod spec renders it in both modes (one template,
    # outside any mode branch), the guardrail validates it, and the NOTES
    # warning's threshold is the same budget the test derives.
    deployment = (HELM_CHART / "templates" / "deployment.yaml").read_text()
    assert (
        'terminationGracePeriodSeconds: {{ include "llm-redact.terminationGracePeriodSeconds" . }}'
        in deployment
    )
    helpers = (HELM_CHART / "templates" / "_helpers.tpl").read_text()
    assert "terminationGracePeriodSeconds must be a non-negative integer" in helpers
    (budget,) = re.findall(
        r'define "llm-redact.shutdownBudgetSeconds" -}}\s*(\d+)\s*{{- end', helpers
    )
    assert int(budget) == _shutdown_budget_seconds()
    notes = (HELM_CHART / "templates" / "NOTES.txt").read_text()
    assert 'include "llm-redact.shutdownBudgetSeconds"' in notes
    resilience = (DEPLOY.parent / "docs" / "resilience.md").read_text()
    assert "terminationGracePeriodSeconds" in resilience
    assert f"defaults to {_default_grace_seconds()} s" in resilience


def _grace_fallback_seconds() -> int:
    helpers = (HELM_CHART / "templates" / "_helpers.tpl").read_text()
    (value,) = re.findall(
        r'define "llm-redact.terminationGracePeriodSeconds" -}}.*?kindIs "invalid" \$grace -}}\s*'
        r"(\d+)\s*{{- else",
        helpers,
        re.DOTALL,
    )
    return int(value)


def test_helm_grace_period_fallback_is_the_chart_default() -> None:
    # An ABSENT value (`helm upgrade --reuse-values` from a release made
    # before it existed) renders the chart default, never Kubernetes' 30 s.
    assert _grace_fallback_seconds() == _default_grace_seconds()


def _chart_copy(tmp_path: Path, values: str | None = None) -> Path:
    chart = tmp_path / "chart"
    shutil.copytree(HELM_CHART, chart)
    if values is not None:
        (chart / "values.yaml").write_text(values)
    # `helm template` does not render NOTES.txt: render a copy of it as the
    # value of a ConfigMap through `tpl`, with the chart's own helpers.
    (chart / "notes.src").write_text((chart / "templates" / "NOTES.txt").read_text())
    (chart / "templates" / "zz-notes.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata: { name: rendered-notes }\n"
        'data:\n  notes: {{ tpl (.Files.Get "notes.src") . | quote }}\n'
    )
    return chart


def _render_copy(chart: Path, *set_args: str) -> list[dict[str, object]]:
    cmd = ["helm", "template", "rel", str(chart)]
    for kv in set_args:
        cmd += ["--set", kv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _copy_grace_and_notes(chart: Path, *set_args: str) -> tuple[object, str]:
    docs = _render_copy(chart, *set_args)
    [deployment] = [d for d in docs if d["kind"] == "Deployment"]
    [notes] = [
        d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "rendered-notes"
    ]
    return deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"], notes["data"][
        "notes"
    ]


@_needs_helm
def test_helm_reuse_values_from_an_older_release_renders_the_default(tmp_path: Path) -> None:
    # `helm upgrade --reuse-values` hands the new chart the OLD release's
    # values, which carry no terminationGracePeriodSeconds: the upgrade must
    # render (it once failed the guardrail) and with the chart default.
    old = (HELM_CHART / "values.yaml").read_text()
    old = re.sub(r"^terminationGracePeriodSeconds:.*\n", "", old, flags=re.MULTILINE)
    assert "terminationGracePeriodSeconds:" not in old
    grace, notes = _copy_grace_and_notes(_chart_copy(tmp_path, old))
    assert grace == _default_grace_seconds()
    assert "WARNING: terminationGracePeriodSeconds" not in notes
    # An explicit null is the same absence.
    assert _pod_spec("terminationGracePeriodSeconds=null")["terminationGracePeriodSeconds"] == 90


@_needs_helm
@pytest.mark.parametrize(
    ("grace", "warns"), [(30, True), (56, True), (0, True), (57, False), (90, False)]
)
def test_helm_notes_warn_below_the_shutdown_budget(tmp_path: Path, grace: int, warns: bool) -> None:
    rendered, notes = _copy_grace_and_notes(
        _chart_copy(tmp_path), f"terminationGracePeriodSeconds={grace}"
    )
    assert rendered == grace
    assert (f"WARNING: terminationGracePeriodSeconds={grace} is below" in notes) is warns


def test_shutdown_budget_docs_name_the_telemetry_tail_outside_it() -> None:
    # The telemetry exporters flush AFTER the vault close (lifespan order),
    # so "at most 57 s" holds only up to the vault close: the docs must say
    # the OTel flush is outside the budget.
    lifespan = inspect.getsource(proxy)
    assert lifespan.index("state.vault_manager.close()") < lifespan.index(
        "state.telemetry.shutdown()"
    )
    resilience = " ".join((DEPLOY.parent / "docs" / "resilience.md").read_text().split())
    assert "The budget ends at the vault close" in resilience
    assert "is outside the 57 s" in resilience
    values = " ".join((HELM_CHART / "values.yaml").read_text().replace("#", " ").split())
    assert "telemetry flush runs after the vault close and is outside those 57 s" in values


def test_shutdown_budget_docs_name_dockers_stop_timeout() -> None:
    # Docker's default stop timeout (10 s) is shorter still than Kubernetes'
    # 30 s: the docs that advise a supervisor stop timeout must name it, and
    # the documented `docker run` must set one covering the budget.
    resilience = " ".join((DEPLOY.parent / "docs" / "resilience.md").read_text().split())
    assert "--stop-timeout 90" in resilience
    assert "stop_grace_period: 90s" in resilience
    deployment = (DEPLOY.parent / "docs" / "deployment.md").read_text()
    (stop,) = re.findall(r"^\s*--stop-timeout (\d+) \\$", deployment, re.MULTILINE)
    assert int(stop) >= _shutdown_budget_seconds()
    assert "stop_grace_period: 90s" in " ".join(deployment.split())


def test_k8s_sidecar_sets_the_charts_grace_period() -> None:
    # The plain manifest is the same pod as the chart's sidecar preset: it
    # sets the same grace period, which covers the shutdown budget.
    doc = next(yaml.safe_load_all((DEPLOY / "k8s-sidecar.yaml").read_text()))
    grace = doc["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    assert grace == _default_grace_seconds()
    assert grace >= _shutdown_budget_seconds()


def _pod_spec(*set_args: str) -> dict[str, object]:
    result = _helm_template(*set_args)
    assert result.returncode == 0, result.stderr
    [deployment] = [d for d in yaml.safe_load_all(result.stdout) if d and d["kind"] == "Deployment"]
    spec: dict[str, object] = deployment["spec"]["template"]["spec"]
    return spec


@_needs_helm
@pytest.mark.parametrize(
    "preset",
    [
        (),
        ("tool.enabled=true",),
        ("mode=standalone", "vault.backend=postgresql"),
        ("mode=standalone", "autoscaling.enabled=true", "vault.backend=postgresql"),
    ],
)
def test_helm_renders_the_grace_period_in_both_modes(preset: tuple[str, ...]) -> None:
    grace = _pod_spec(*preset)["terminationGracePeriodSeconds"]
    assert grace == _default_grace_seconds()
    assert isinstance(grace, int) and grace >= _shutdown_budget_seconds()


@_needs_helm
def test_helm_grace_period_is_overridable() -> None:
    assert _pod_spec("terminationGracePeriodSeconds=120")["terminationGracePeriodSeconds"] == 120
    # 0 is a legal (if unwise) Kubernetes value: allowed, NOTES warns.
    assert _pod_spec("terminationGracePeriodSeconds=0")["terminationGracePeriodSeconds"] == 0


@_needs_helm
@pytest.mark.parametrize(
    "flag",
    [
        ("--set", "terminationGracePeriodSeconds=-1"),
        ("--set", "terminationGracePeriodSeconds=1.5"),
        ("--set", "terminationGracePeriodSeconds=abc"),
        ("--set", "terminationGracePeriodSeconds=true"),
        ("--set-string", "terminationGracePeriodSeconds=90"),
    ],
)
def test_helm_grace_period_rejects_a_non_integer(flag: tuple[str, str]) -> None:
    # A quoted or fractional value would only fail at apply time: fail the render.
    result = subprocess.run(
        ["helm", "template", "rel", str(HELM_CHART), *flag], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "non-negative integer" in result.stderr


# --- Helm standalone preStop delay (endpoint removal before SIGTERM) ---------


def _default_prestop_seconds() -> int:
    text = (HELM_CHART / "values.yaml").read_text()
    (value,) = re.findall(r"^preStopSleepSeconds:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(value)


def test_helm_prestop_default_fits_inside_the_grace_period() -> None:
    # The preStop delay runs BEFORE SIGTERM and counts against the grace
    # period: the defaults must still leave the whole shutdown budget.
    helpers = (HELM_CHART / "templates" / "_helpers.tpl").read_text()
    (fallback,) = re.findall(
        r'define "llm-redact.preStopSleepSeconds" -}}.*?kindIs "invalid" \$preStop -}}\s*'
        r"(\d+)\s*{{- else",
        helpers,
        re.DOTALL,
    )
    assert int(fallback) == _default_prestop_seconds() == 5
    assert _default_grace_seconds() >= _default_prestop_seconds() + _shutdown_budget_seconds()
    assert "preStopSleepSeconds must be a non-negative integer" in helpers


def _proxy_prestop(*set_args: str) -> object:
    spec = _pod_spec(*set_args)
    containers = spec["containers"]
    assert isinstance(containers, list)
    [proxy_container] = [c for c in containers if c["name"] == "llm-redact"]
    lifecycle = proxy_container.get("lifecycle")
    return None if lifecycle is None else lifecycle["preStop"]["exec"]["command"]


_STANDALONE = ("mode=standalone", "vault.backend=postgresql")


@_needs_helm
def test_helm_standalone_renders_a_prestop_delay_sidecar_none() -> None:
    sleep = ["python", "-c", "import time; time.sleep(5)"]
    assert _proxy_prestop(*_STANDALONE) == sleep
    assert _proxy_prestop(*_STANDALONE, "autoscaling.enabled=true") == sleep
    # Sidecar: the tool dials loopback and there is no Service to drain.
    assert _proxy_prestop() is None
    assert _proxy_prestop("tool.enabled=true") is None
    assert _proxy_prestop("preStopSleepSeconds=9") is None


@_needs_helm
def test_helm_prestop_is_overridable_and_absent_means_default() -> None:
    assert _proxy_prestop(*_STANDALONE, "preStopSleepSeconds=12") == [
        "python",
        "-c",
        "import time; time.sleep(12)",
    ]
    assert _proxy_prestop(*_STANDALONE, "preStopSleepSeconds=0") is None
    assert _proxy_prestop(*_STANDALONE, "preStopSleepSeconds=null") == [
        "python",
        "-c",
        "import time; time.sleep(5)",
    ]


@_needs_helm
@pytest.mark.parametrize("value", ["-1", "1.5", "abc", "true"])
def test_helm_prestop_rejects_a_non_integer(value: str) -> None:
    result = _helm_template(*_STANDALONE, f"preStopSleepSeconds={value}")
    assert result.returncode != 0
    assert "preStopSleepSeconds must be a non-negative integer" in result.stderr


@_needs_helm
def test_helm_reuse_values_from_an_older_release_renders_the_prestop(tmp_path: Path) -> None:
    old = (HELM_CHART / "values.yaml").read_text()
    old = re.sub(
        r"^(terminationGracePeriodSeconds|preStopSleepSeconds):.*\n", "", old, flags=re.MULTILINE
    )
    assert "preStopSleepSeconds:" not in old
    docs = _render_copy(_chart_copy(tmp_path, old), *_STANDALONE)
    [deployment] = [d for d in docs if d["kind"] == "Deployment"]
    spec = deployment["spec"]["template"]["spec"]
    [proxy_container] = [c for c in spec["containers"] if c["name"] == "llm-redact"]
    assert (
        proxy_container["lifecycle"]["preStop"]["exec"]["command"][-1]
        == "import time; time.sleep(5)"
    )
    assert spec["terminationGracePeriodSeconds"] == 90


def test_helm_templates_read_the_models_table_nil_safely() -> None:
    # Stdlib needle (no helm needed): an older release's values carry no
    # `models` table, and `.Values.models.volume` on it fails every render.
    templates = HELM_CHART / "templates"
    for path in [*templates.glob("*.tpl"), *templates.glob("*.yaml"), templates / "NOTES.txt"]:
        assert ".Values.models.volume" not in path.read_text(), path.name


@_needs_helm
@pytest.mark.parametrize("mode", ["sidecar", "standalone"])
def test_helm_reuse_values_from_an_older_release_renders_without_models(
    tmp_path: Path, mode: str
) -> None:
    # `helm upgrade --reuse-values` from a release made before the models
    # table: no `models` and no `image.variant` in the old values.
    old = (HELM_CHART / "values.yaml").read_text()
    old = re.sub(r"^models:\n(?:  .*\n)+", "", old, flags=re.MULTILINE)
    old = re.sub(r"^  variant:.*\n", "", old, flags=re.MULTILINE)
    assert "\nmodels:" not in old and "variant:" not in old
    docs = _render_copy(_chart_copy(tmp_path, old), *(_STANDALONE if mode == "standalone" else ()))
    [deployment] = [d for d in docs if d["kind"] == "Deployment"]
    spec = deployment["spec"]["template"]["spec"]
    assert all(volume["name"] != "models" for volume in spec["volumes"])
    [proxy_container] = [c for c in spec["containers"] if c["name"] == "llm-redact"]
    assert all(mount["mountPath"] != "/models" for mount in proxy_container["volumeMounts"])
    assert all(env["name"] != "HF_HUB_OFFLINE" for env in proxy_container["env"])
    [notes] = [
        d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "rendered-notes"
    ]
    assert "NER models" not in notes["data"]["notes"]
    # An explicit null is the same absence.
    assert _render_copy(_chart_copy(tmp_path / "null"), "models=null")


@_needs_helm
def test_helm_notes_count_the_prestop_delay_in_standalone_only(tmp_path: Path) -> None:
    chart = _chart_copy(tmp_path)
    # 60 covers the 57 s budget, but not the budget + the 5 s preStop delay.
    _, notes = _copy_grace_and_notes(chart, *_STANDALONE, "terminationGracePeriodSeconds=60")
    assert "WARNING: terminationGracePeriodSeconds=60 is below" in notes
    assert "preStop delay (5 s" in " ".join(notes.split())
    _, notes = _copy_grace_and_notes(chart, "terminationGracePeriodSeconds=60")
    assert "WARNING: terminationGracePeriodSeconds" not in notes
    _, notes = _copy_grace_and_notes(
        chart, *_STANDALONE, "terminationGracePeriodSeconds=60", "preStopSleepSeconds=0"
    )
    assert "WARNING: terminationGracePeriodSeconds" not in notes
    _, notes = _copy_grace_and_notes(chart, *_STANDALONE)
    assert "WARNING: terminationGracePeriodSeconds" not in notes


def test_the_image_ships_the_extras_its_features_need() -> None:
    # Without extract, [extraction] with its default formats (pdf included)
    # refuses to start inside the image.
    dockerfile = (DEPLOY.parent / "Dockerfile").read_text()
    (default,) = re.findall(r'^ARG EXTRAS="([^"]*)"$', dockerfile, re.MULTILINE)
    assert set(re.findall(r"--extra (\S+)", default)) == {"perf", "realtime", "extract"}


def test_the_image_license_label_is_the_declared_license() -> None:
    # The OCI label still said MIT after the relicensing to AGPL-3.0-only;
    # every image label must name what pyproject.toml declares.
    root = DEPLOY.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["license"]
    dockerfile = (root / "Dockerfile").read_text()
    labels = re.findall(r'org\.opencontainers\.image\.licenses="([^"]*)"', dockerfile)
    assert labels == [declared]
    # The published image takes its labels from docker/metadata-action, whose
    # default licence label is the repository's GitHub-detected SPDX id
    # ("AGPL-3.0"), overriding the Dockerfile's: release.yml pins it.
    release = (root / ".github" / "workflows" / "release.yml").read_text()
    pinned = re.findall(r"org\.opencontainers\.image\.licenses=(\S+)", release)
    assert pinned == [declared]


# --- the license text in the images ---------------------------------------------
# The images once shipped no copy of the AGPL text: no step put it at a doc
# path, and the builder never saw LICENSE, so the wheel built there lacked
# licenses/LICENSE.

ROOT = DEPLOY.parent


def test_the_dockerfile_copies_the_license_into_both_stages() -> None:
    # The builder copies it after the dependency-install RUN (which it would
    # otherwise re-run whenever the text changes) and before the RUN that
    # builds the project's wheel; the runtime stage puts it at the doc path.
    lines = (ROOT / "Dockerfile").read_text().splitlines()

    def where(text: str) -> list[int]:
        return [i for i, line in enumerate(lines) if text in line]

    copies = [i for i in where("LICENSE") if lines[i].startswith("COPY")]
    assert [lines[i] for i in copies] == [
        "COPY LICENSE ./",
        "COPY LICENSE /usr/share/doc/llm-redact/LICENSE",
    ]
    [dependencies] = where("uv sync --frozen --no-dev --no-install-project")
    [project] = where("uv sync --frozen --no-dev --no-editable")
    [runtime] = where("FROM python:3.13-slim-bookworm AS runtime")
    # The doc copy belongs to the runtime stage itself (release.yml builds
    # that target), not to a later stage built FROM it.
    runtime_end = next(
        (i for i in range(runtime + 1, len(lines)) if lines[i].startswith("FROM ")), len(lines)
    )
    assert dependencies < copies[0] < project < runtime < copies[1] < runtime_end


SMOKE_LICENSE_CHECK = (
    '"$ENGINE" exec -i smoke-proxy python - < "$(dirname "$0")/image_license_check.py" \\\n'
    '  || { echo "FAIL: the AGPL-3.0 license text is missing from the image"; exit 1; }\n'
)
NER_LICENSE_CHECK = (
    "docker run --rm --network none --entrypoint python \\\n"
    '  -v "$PWD/scripts/image_license_check.py:/smoke/image_license_check.py:ro" \\\n'
    "  llm-redact:ner-ci /smoke/image_license_check.py\n"
)


def test_both_images_run_the_license_check() -> None:
    # The stock image in the container smoke test, whose handler fails it
    # when the check fails; the -ner image in a container-ner step, which no
    # continue-on-error lets pass when the check fails.
    smoke = (ROOT / "scripts" / "container_smoke.sh").read_text()
    assert f"\n{SMOKE_LICENSE_CHECK}" in smoke
    job = _workflow("ci.yml")["jobs"]["container-ner"]
    [step] = [step for step in job["steps"] if step.get("run") == NER_LICENSE_CHECK]
    assert "continue-on-error" not in step
    assert "continue-on-error" not in job


def _license_check():
    path = ROOT / "scripts" / "image_license_check.py"
    spec = importlib.util.spec_from_file_location("image_license_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_image_license_check_names_what_is_wrong() -> None:
    check = _license_check()
    # The path is pinned once; the messages render it as the host does (the
    # suite also runs on Windows, where it is written with backslashes).
    assert check.DOC.as_posix() == "/usr/share/doc/llm-redact/LICENSE"
    doc = check.DOC
    agpl = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert check.problems(agpl, agpl) == []
    assert check.problems(None, agpl) == [f"{doc} is missing"]
    assert check.problems("MIT License", agpl) == [f"{doc} does not hold the AGPL-3.0 text"]
    metadata = "the llm-redact-proxy distribution"
    assert check.problems(agpl, None) == [f"{metadata} carries no licenses/LICENSE"]
    assert check.problems(agpl, "MIT License") == [
        f"{metadata}'s licenses/LICENSE does not hold the AGPL-3.0 text"
    ]
    assert check.problems(agpl, agpl + "\n") == [f"{metadata}'s licenses/LICENSE is not {doc}"]


def test_the_image_license_check_fails_without_the_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    check = _license_check()
    (tmp_path / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
    monkeypatch.setattr(check, "DOC", tmp_path / "LICENSE")

    def not_installed(name: str) -> None:
        raise check.PackageNotFoundError(name)

    monkeypatch.setattr(check, "distribution", not_installed)
    assert check.main() == 1
    assert capsys.readouterr().err == "FAIL: the llm-redact-proxy distribution is not installed\n"


# --- the -ner image variant and the air-gap CI jobs (static: CI builds them) -----

WORKFLOWS = DEPLOY.parent / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _extras(flags: str) -> set[str]:
    return set(re.findall(r"--extra (\S+)", flags))


def test_the_ner_image_is_built_from_the_same_values_everywhere() -> None:
    dockerfile = (DEPLOY.parent / "Dockerfile").read_text()
    (stock,) = re.findall(r'^ARG EXTRAS="([^"]*)"$', dockerfile, re.MULTILINE)
    assert re.findall(r'^ARG NER_EXTRAS="([^"]*)"$', dockerfile, re.MULTILINE) == [""]
    assert re.findall(r'^ARG TORCH_INDEX_URL="([^"]*)"$', dockerfile, re.MULTILINE) == [
        "https://download.pytorch.org/whl/cpu"
    ]
    release = _workflow("release.yml")
    ci = _workflow("ci.yml")
    # The SBOM describes the image's closure: the stock extras plus the NER ones.
    assert _extras(release["env"]["IMAGE_EXTRAS"]) == _extras(stock)
    ner = _extras(release["env"]["NER_EXTRAS"])
    assert ner == {"hf", "gliner"}
    assert ner <= set(
        tomllib.loads((DEPLOY.parent / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
    )
    # CI builds and smoke-tests exactly what a release builds.
    assert _extras(ci["jobs"]["container-ner"]["env"]["NER_EXTRAS"]) == ner


def test_the_release_publishes_and_signs_both_images() -> None:
    job = _workflow("release.yml")["jobs"]["publish-ghcr"]
    variants = {row["variant"]: row for row in job["strategy"]["matrix"]["include"]}
    assert set(variants) == {"stock", "ner"}
    assert (variants["stock"]["suffix"], variants["ner"]["suffix"]) == ("", "-ner")
    assert (variants["stock"]["ner"], variants["ner"]["ner"]) == (False, True)
    assert job["strategy"]["fail-fast"] is False
    steps = {step.get("id") or step.get("name") or step.get("uses"): step for step in job["steps"]}
    meta = steps["meta"]["with"]
    assert meta["flavor"].strip() == "suffix=${{ matrix.suffix }},onlatest=true"
    build = steps["build"]["with"]
    assert build["platforms"] == "linux/amd64,linux/arm64"
    assert build["sbom"] is True
    assert "NER_EXTRAS=${{ matrix.ner && env.NER_EXTRAS || '' }}" in build["build-args"]
    assert (
        'cosign sign --yes "ghcr.io/${GITHUB_REPOSITORY}@${DIGEST}"'
        in (steps["sign the image"]["run"])
    )
    sbom = _workflow("release.yml")["jobs"]["build"]["steps"]
    assert any("llm-redact-ner-image.cdx.json" in step.get("run", "") for step in sbom)


def test_the_cpu_torch_recipe_guards_the_ner_image_build() -> None:
    dockerfile = (DEPLOY.parent / "Dockerfile").read_text()
    assert "COPY scripts/cpu_torch.py /tmp/cpu_torch.py" in dockerfile
    assert "python /tmp/cpu_torch.py requirements /tmp/requirements.txt" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "/app/.venv/bin/python /tmp/cpu_torch.py check" in dockerfile
    # The stock image keeps its frozen uv sync.
    assert "uv sync --frozen --no-dev --no-editable ${EXTRAS}" in dockerfile
    # The models cache sits on the writable /data volume.
    assert "HF_HOME=/data/huggingface" in dockerfile


def _torch_installs(text: str) -> list[str]:
    """Every command of ``text`` (backslash continuations joined, ``&&``
    chains split) that downloads or installs from a ``torch.txt``."""
    joined = re.sub(r"\\\n\s*", " ", text)
    commands = [part for line in joined.splitlines() for part in line.split("&&")]
    return [
        command
        for command in commands
        if "torch.txt" in command and re.search(r"\bpip\"? (?:install|download)\b", command)
    ]


@pytest.mark.parametrize(
    ("path", "count"),
    [
        ("Dockerfile", 1),
        (".github/workflows/ci.yml", 2),
        ("tests/airgap/install_wheelhouse.sh", 1),
        ("scripts/ner_ci_env.sh", 1),
        ("docs/deployment.md", 2),
    ],
)
def test_torch_is_installed_hash_checked_everywhere(path: str, count: int) -> None:
    # torch.txt carries the CPU wheels' SHA-256 (scripts/cpu_torch.py); a
    # torch download or install without --require-hashes would take any wheel
    # named like the locked version.
    installs = _torch_installs((DEPLOY.parent / path).read_text())
    assert len(installs) == count, installs
    for command in installs:
        assert "--require-hashes" in command, command


def test_the_ner_sbom_records_torch_with_its_digests() -> None:
    runs = "\n".join(
        step.get("run", "") for step in _workflow("release.yml")["jobs"]["build"]["steps"]
    )
    assert "cat torch.txt >>sbom-ner-requirements.txt" in runs
    assert '+cpu" >>' not in runs


def _pinned(steps: list[dict]) -> None:
    for step in steps:
        uses = step.get("uses")
        if uses is None:
            continue
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", uses), uses
        if uses.startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False


@pytest.mark.parametrize("job", ["airgap", "container-ner"])
def test_the_airgap_jobs_pin_their_actions(job: str) -> None:
    _pinned(_workflow("ci.yml")["jobs"][job]["steps"])


def test_the_airgap_job_runs_offline_without_a_route() -> None:
    steps = _workflow("ci.yml")["jobs"]["airgap"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "sudo unshare --net --mount" in runs
    assert "tests/airgap/run_offline.sh" in runs
    assert "sudo unshare --net -- " in runs and "tests/airgap/install_wheelhouse.sh" in runs
    for config in ("pull.toml", "pull-edge.toml"):
        assert f"--config tests/airgap/{config}" in runs
    offline = (DEPLOY.parent / "tests" / "airgap" / "run_offline.sh").read_text()
    for needle in (
        "no_network.py",
        "models verify --dir /models",
        "serve --check --config tests/airgap/config.toml",
        "serve --check --config tests/airgap/edge.toml",
        "preview --config tests/airgap/presidio.toml",
        "llm-redact models pull --to DIR",
        "mount -o remount,bind,ro /models",
    ):
        assert needle in offline


# --- the models volume and the -ner image (T17e; docs/air-gapped.md) -------------

_MODELS_PVC = (
    "models.volume.persistentVolumeClaim.claimName=llm-redact-models",
    "models.volume.persistentVolumeClaim.readOnly=true",
)


def _proxy_container(docs: list[dict]) -> dict:
    [deployment] = [d for d in docs if d["kind"] == "Deployment"]
    pod = deployment["spec"]["template"]["spec"]
    [proxy_container] = [c for c in pod["containers"] if c["name"] == "llm-redact"]
    return {"pod": pod, "container": proxy_container}


def test_the_models_volume_values_default_off() -> None:
    values = yaml.safe_load((HELM_CHART / "values.yaml").read_text())
    assert values["image"]["variant"] == ""
    assert values["models"] == {"volume": {}}


@_needs_helm
@pytest.mark.parametrize("preset", [(), _STANDALONE], ids=["sidecar", "standalone"])
def test_helm_models_volume_mounts_read_only_and_goes_offline(preset: tuple[str, ...]) -> None:
    result = _helm_template(*preset, *_MODELS_PVC, "image.variant=ner")
    assert result.returncode == 0, result.stderr
    found = _proxy_container([d for d in yaml.safe_load_all(result.stdout) if d])
    container, pod = found["container"], found["pod"]
    assert container["image"] == f"ghcr.io/asanderson/llm-redact:{__version__}-ner"
    assert {"name": "models", "mountPath": "/models", "readOnly": True} in container["volumeMounts"]
    [volume] = [v for v in pod["volumes"] if v["name"] == "models"]
    assert volume == {
        "name": "models",
        "persistentVolumeClaim": {"claimName": "llm-redact-models", "readOnly": True},
    }
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    # The hardened container spec is unchanged: the root stays read-only.
    assert container["securityContext"]["readOnlyRootFilesystem"] is True


@_needs_helm
def test_helm_without_a_models_volume_renders_neither_mount_nor_offline_env() -> None:
    result = _helm_template()
    assert result.returncode == 0, result.stderr
    found = _proxy_container([d for d in yaml.safe_load_all(result.stdout) if d])
    container, pod = found["container"], found["pod"]
    assert container["image"] == f"ghcr.io/asanderson/llm-redact:{__version__}"
    assert all(m["name"] != "models" for m in container["volumeMounts"])
    assert all(v["name"] != "models" for v in pod["volumes"])
    names = {e["name"] for e in container["env"]}
    assert not names & {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}


@_needs_helm
def test_helm_image_variant_keeps_an_explicit_tag() -> None:
    result = _helm_template("image.tag=1.2.3", "image.variant=ner")
    assert result.returncode == 0, result.stderr
    found = _proxy_container([d for d in yaml.safe_load_all(result.stdout) if d])
    assert found["container"]["image"] == "ghcr.io/asanderson/llm-redact:1.2.3-ner"


@_needs_helm
@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("image.variant=gpu", 'image.variant must be "" (the stock image) or "ner"'),
        ("models.volume=llm-redact-models", "models.volume must be a volume source"),
    ],
)
def test_helm_rejects_a_bad_variant_or_models_volume(flag: str, message: str) -> None:
    result = _helm_template(flag)
    assert result.returncode != 0
    assert message in result.stderr


@_needs_helm
@pytest.mark.parametrize(("variant", "warns"), [("ner", False), ("", True)])
def test_helm_notes_name_the_models_volume_and_the_stock_image_gap(
    tmp_path: Path, variant: str, warns: bool
) -> None:
    chart = _chart_copy(tmp_path)
    _, notes = _copy_grace_and_notes(chart, *_MODELS_PVC, f"image.variant={variant}")
    assert "llm-redact models verify --dir /models" in str(notes)
    assert ("models.volume is set but image.variant is not" in str(notes)) is warns


AIR_GAPPED = DEPLOY.parent / "docs" / "air-gapped.md"


def test_the_air_gapped_pull_never_opens_the_models_folder_to_everyone() -> None:
    # The manifest is written into the folder its SHA-256 is noted from: a
    # folder other users can write lets them change a file AND the manifest
    # before the hash is noted. The pull runs as the operator instead.
    guide = AIR_GAPPED.read_text()
    assert not re.search(r"\b0?77[67]\b|a\+w|o\+w", guide)
    connected = guide[guide.index("## On the connected machine") : guide.index("## Inside")]
    assert "mkdir -m 0755 models" in connected
    pull = re.sub(r"\\\n\s*", " ", connected)
    [command] = [line for line in pull.splitlines() if "models pull --to /out" in line]
    assert '--user "$(id -u):$(id -g)"' in command and "-e HF_HOME=/tmp/hf" in command


def test_the_air_gapped_guide_never_claims_an_egress_filter_in_the_user_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `service install` writes a systemd USER unit, whose manager cannot attach
    # the cgroup BPF programs IPAddressDeny= needs: systemd ignores it there.
    from llm_redact import service_cli

    monkeypatch.setattr(service_cli.sys, "platform", "linux")
    assert "systemd/user" in service_cli._unit_path().as_posix()
    guide = AIR_GAPPED.read_text()
    systemd = guide[guide.index("## systemd (native installs)") : guide.index("## Troubleshooting")]
    assert "systemctl --user edit" not in systemd
    assert "has **no effect** there" in systemd
    assert "not running as root." in systemd
    assert "meta skuid" in systemd and "/etc/systemd/system/" in systemd
    # The IPAddressDeny= example sits under the system-unit option only.
    assert systemd.index("IPAddressDeny=any") > systemd.index("A system unit you write yourself")


def test_the_network_wrapper_claims_only_what_an_audit_hook_sees() -> None:
    # A Python audit hook sees the socket module's calls only; native code
    # (hf-xet's Rust client) resolves and connects past it.
    wrapper = (DEPLOY.parent / "tests" / "airgap" / "no_network.py").read_text()
    assert "through\nPython's ``socket`` module" in wrapper
    assert "invisible to an audit\nhook" in wrapper
    for path in ("docs/air-gapped.md", "docs/privacy.md", "CHANGELOG.md"):
        text = re.sub(r"\s+", " ", (DEPLOY.parent / path).read_text())
        assert "attempt made through Python's `socket` module" in text, path
        assert "fails on any attempt to resolve a host or open a connection" not in text, path
