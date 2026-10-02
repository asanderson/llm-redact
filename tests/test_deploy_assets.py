"""The deploy/ ops assets are well-formed and reference only real metrics.

A Grafana panel or an alert rule that names a metric the proxy does not emit
is silently broken — it just shows "No data". This test parses the assets and
asserts every `llm_redact_*` token they reference is an actual metric (allowing
the histogram `_bucket`/`_sum`/`_count` suffixes), so renaming a metric without
updating the dashboard fails CI.
"""

import json
import math
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import httpx
import pytest
import yaml

from llm_redact import __version__
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


@pytest.mark.anyio
async def test_deploy_assets_reference_only_real_metrics() -> None:
    emitted = await _emitted_metric_names()
    for asset in ("grafana-dashboard.json", "prometheus-alerts.yml"):
        referenced = _referenced_metrics((DEPLOY / asset).read_text())
        assert referenced, f"{asset} references no llm_redact metrics"
        unknown = {m for m in referenced if _canonical(m) not in emitted}
        assert not unknown, f"{asset} references metrics the proxy does not emit: {sorted(unknown)}"


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
        "terminationGracePeriodSeconds: {{ int64 .Values.terminationGracePeriodSeconds }}"
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
        ("--set", "terminationGracePeriodSeconds=null"),
        ("--set-string", "terminationGracePeriodSeconds=90"),
    ],
)
def test_helm_grace_period_rejects_a_non_integer(flag: tuple[str, str]) -> None:
    # A quoted or fractional value would only fail at apply time, and a
    # deleted one would silently fall back to Kubernetes' 30 s: fail the render.
    result = subprocess.run(
        ["helm", "template", "rel", str(HELM_CHART), *flag], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "non-negative integer" in result.stderr


def test_the_image_ships_the_extras_its_features_need() -> None:
    # Without extract, [extraction] with its default formats (pdf included)
    # refuses to start inside the image.
    dockerfile = (DEPLOY.parent / "Dockerfile").read_text()
    (default,) = re.findall(r'^ARG EXTRAS="([^"]*)"$', dockerfile, re.MULTILINE)
    assert set(re.findall(r"--extra (\S+)", default)) == {"perf", "realtime", "extract"}
