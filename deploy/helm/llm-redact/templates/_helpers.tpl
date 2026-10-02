{{/*
Chart name / fullname / labels — the standard Helm helpers.
*/}}
{{- define "llm-redact.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llm-redact.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "llm-redact.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "llm-redact.labels" -}}
app.kubernetes.io/name: {{ include "llm-redact.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "llm-redact.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llm-redact.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "llm-redact.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}

{{/*
GUARDRAIL — the load-bearing never-wrong-value check.
A standalone proxy that autoscales (or runs >1 replica) MUST use a shared server
vault: with a per-pod memory/sqlite vault, replicas would issue divergent
«TYPE_NNN» tokens and one pod could rehydrate another's secret. Fail the render.
*/}}
{{- define "llm-redact.validate" -}}
{{- $grace := .Values.terminationGracePeriodSeconds -}}
{{- if not (kindIs "invalid" $grace) -}}
{{- if not (or (kindIs "int" $grace) (kindIs "int64" $grace) (kindIs "float64" $grace)) -}}
{{- fail "llm-redact: terminationGracePeriodSeconds must be a non-negative integer (seconds) — the proxy's shutdown budget; see values.yaml." -}}
{{- end -}}
{{- if or (lt (float64 $grace) 0.0) (ne (float64 $grace) (float64 (int64 $grace))) -}}
{{- fail "llm-redact: terminationGracePeriodSeconds must be a non-negative integer (seconds) — the proxy's shutdown budget; see values.yaml." -}}
{{- end -}}
{{- end -}}
{{- $preStop := .Values.preStopSleepSeconds -}}
{{- if not (kindIs "invalid" $preStop) -}}
{{- if not (or (kindIs "int" $preStop) (kindIs "int64" $preStop) (kindIs "float64" $preStop)) -}}
{{- fail "llm-redact: preStopSleepSeconds must be a non-negative integer (seconds) — the standalone pod's delay before SIGTERM; see values.yaml." -}}
{{- end -}}
{{- if or (lt (float64 $preStop) 0.0) (ne (float64 $preStop) (float64 (int64 $preStop))) -}}
{{- fail "llm-redact: preStopSleepSeconds must be a non-negative integer (seconds) — the standalone pod's delay before SIGTERM; see values.yaml." -}}
{{- end -}}
{{- end -}}
{{- if eq .Values.mode "standalone" -}}
{{- $multi := or .Values.autoscaling.enabled (gt (int .Values.replicaCount) 1) -}}
{{- if and $multi (or (eq .Values.vault.backend "memory") (eq .Values.vault.backend "sqlite")) -}}
{{- fail "llm-redact: a standalone autoscaled/multi-replica proxy needs a SHARED vault (vault.backend must be postgresql/mysql/oracle/dbapi) — a per-pod memory/sqlite vault would issue inconsistent tokens across replicas (never-wrong-value). Set vault.backend to a server backend, or set autoscaling.enabled=false and replicaCount=1." -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
SHUTDOWN BUDGET — the most the proxy's BOUNDED shutdown steps take after its
in-flight requests have finished (docs/resilience.md, "Shutdown order"):
SHUTDOWN_DRAIN_SECONDS (5) + _SINK_CLOSE_TIMEOUT_SECONDS (45) +
_SINK_CANCEL_GRACE_SECONDS (1) + the vault close's second map drain
SHUTDOWN_DRAIN_SECONDS (5) + STOP_JOIN_SECONDS (1). NOTES.txt warns when
terminationGracePeriodSeconds is below it; test_deploy_assets.py recomputes
it from those constants, so changing one fails the test until this follows.
*/}}
{{- define "llm-redact.shutdownBudgetSeconds" -}}
57
{{- end -}}

{{/*
The pod's terminationGracePeriodSeconds: the value (validated above), or the
chart default when the key is ABSENT — `helm upgrade --reuse-values` from a
release made before the value existed carries no such key, and neither does
`--set terminationGracePeriodSeconds=null`. Never Kubernetes' 30 s, and not
`default`, which would also turn an explicit 0 into 90. The fallback equals
values.yaml's default (test_deploy_assets.py pins both).
*/}}
{{- define "llm-redact.terminationGracePeriodSeconds" -}}
{{- $grace := .Values.terminationGracePeriodSeconds -}}
{{- if kindIs "invalid" $grace -}}
90
{{- else -}}
{{ int64 $grace }}
{{- end -}}
{{- end -}}

{{/*
STANDALONE preStop delay: seconds the proxy container waits, still serving,
between the pod turning Terminating and its SIGTERM, so the Service's
endpoints (kube-proxy, an Ingress) stop sending NEW connections to it first —
without it a rolling update can refuse a few connections while the endpoint
removal propagates. Availability only: the drain itself needs no hook (uvicorn
acts on SIGTERM). The delay counts against terminationGracePeriodSeconds
(NOTES warns when the two leave less than the shutdown budget). Absent (an
older release's values under --reuse-values, or null) = the chart default;
0 renders no hook. Sidecar mode never renders one (the tool dials loopback,
there is no Service to drain).
*/}}
{{- define "llm-redact.preStopSleepSeconds" -}}
{{- $preStop := .Values.preStopSleepSeconds -}}
{{- if kindIs "invalid" $preStop -}}
5
{{- else -}}
{{ int64 $preStop }}
{{- end -}}
{{- end -}}

{{/*
The effective replica count for the standalone Deployment (HPA owns it when on).
*/}}
{{- define "llm-redact.replicas" -}}
{{- if .Values.autoscaling.enabled -}}
{{- .Values.autoscaling.minReplicas -}}
{{- else -}}
{{- .Values.replicaCount -}}
{{- end -}}
{{- end -}}

{{/*
The generated proxy config file (mounted read-only, LLM_REDACT_CONFIG points here).
Secrets (vault key, DSN, license) come from env/Secrets — never this ConfigMap.
*/}}
{{- define "llm-redact.configToml" -}}
{{- /*
allowed_hosts: the host names the proxy answers to besides loopback and its
bind host. A request that spends a credential the proxy holds (auth =
"identity" via IRSA / Workload Identity, a routed operator key) must be
addressed to one of them — the DNS-rebinding defense — so standalone mode
lists the in-cluster names of the Service its clients dial; allowedHosts adds
more (an Ingress host, a custom cluster domain).
*/ -}}
{{- $hosts := .Values.allowedHosts | default list -}}
{{- if eq .Values.mode "standalone" -}}
{{- $svc := include "llm-redact.fullname" . -}}
{{- $ns := .Release.Namespace -}}
{{- $hosts = concat (list $svc (printf "%s.%s" $svc $ns) (printf "%s.%s.svc" $svc $ns) (printf "%s.%s.svc.cluster.local" $svc $ns)) $hosts -}}
{{- end -}}
{{- if $hosts -}}
allowed_hosts = {{ toJson $hosts }}
{{ end -}}
[vault]
backend = {{ .Values.vault.backend | quote }}
{{- if ne .Values.vault.encryption "none" }}
encryption = {{ .Values.vault.encryption | quote }}
{{- end }}
[log]
format = {{ .Values.log.format | quote }}
{{- with .Values.extraConfig }}

{{ . }}
{{- end }}
{{- end -}}

{{/*
The hardened llm-redact proxy container, shared by both modes.

BIND HONESTY (3.3.0): in sidecar mode the proxy binds 127.0.0.1 — containers
in a pod share the network namespace, so the tool reaches it over loopback
while other pods cannot reach it AT ALL (the "never exposed" claim holds with
no NetworkPolicy needed). A loopback bind breaks kubelet httpGet probes
(kubelet dials the POD IP), so sidecar mode uses exec probes via the image's
own Python — the same one-liner as the Dockerfile HEALTHCHECK. Standalone
binds 0.0.0.0 (cross-pod reach is the point) and keeps httpGet probes.
*/}}
{{- define "llm-redact.proxyContainer" -}}
- name: llm-redact
  image: {{ include "llm-redact.image" . | quote }}
  imagePullPolicy: {{ .Values.image.pullPolicy }}
  args: ["serve", "--config", "/etc/llm-redact/config.toml"]
  env:
    {{- if eq .Values.mode "standalone" }}
    - { name: LLM_REDACT_HOST, value: "0.0.0.0" }
    {{- if .Values.insecureBind }}
    - { name: LLM_REDACT_INSECURE_BIND, value: "1" }
    {{- end }}
    {{- else }}
    - { name: LLM_REDACT_HOST, value: "127.0.0.1" }
    {{- end }}
    - { name: LLM_REDACT_PORT, value: "8787" }
    - { name: XDG_DATA_HOME, value: "/data" }
    - name: LLM_REDACT_VAULT_KEY
      valueFrom:
        secretKeyRef: { name: {{ .Values.vault.keySecret | quote }}, key: vault-key, optional: true }
    {{- if .Values.vault.dsnSecret }}
    - name: LLM_REDACT_VAULT_DSN
      valueFrom:
        secretKeyRef: { name: {{ .Values.vault.dsnSecret | quote }}, key: {{ .Values.vault.dsnSecretKey | quote }} }
    {{- end }}
    - name: LLM_REDACT_LICENSE_KEY
      valueFrom:
        secretKeyRef: { name: {{ .Values.license.secretName | quote }}, key: license-key, optional: true }
    {{- with .Values.extraEnv }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  ports:
    - { name: proxy, containerPort: 8787 }
  {{- if eq .Values.mode "standalone" }}
  readinessProbe:
    httpGet: { path: /__llm-redact/readyz, port: 8787 }
    initialDelaySeconds: 2
    periodSeconds: 10
  livenessProbe:
    httpGet: { path: /__llm-redact/healthz, port: 8787 }
    periodSeconds: 20
  {{- $preStop := int64 (include "llm-redact.preStopSleepSeconds" .) }}
  {{- if gt $preStop 0 }}
  # Keep serving while the Service stops routing new connections here
  # (values.yaml preStopSleepSeconds); python, as the exec probes use.
  lifecycle:
    preStop:
      exec:
        command: ["python", "-c", "import time; time.sleep({{ $preStop }})"]
  {{- end }}
  {{- else }}
  readinessProbe:
    exec:
      command: ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8787/__llm-redact/readyz')"]
    initialDelaySeconds: 2
    periodSeconds: 10
  livenessProbe:
    exec:
      command: ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8787/__llm-redact/healthz')"]
    periodSeconds: 20
  {{- end }}
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities: { drop: ["ALL"] }
  resources:
    {{- toYaml .Values.resources | nindent 4 }}
  volumeMounts:
    - { name: config, mountPath: /etc/llm-redact, readOnly: true }
    - { name: redact-data, mountPath: /data }
    {{- with .Values.extraVolumeMounts }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
{{- end -}}

{{/*
Pod volumes shared by both modes. extraVolumes exists so a [tls] cert/key/
client-ca Secret can actually be mounted — without it the chart's own
"prefer mutual TLS" advice was unwireable.
*/}}
{{- define "llm-redact.volumes" -}}
- name: config
  configMap:
    name: {{ include "llm-redact.fullname" . }}-config
- name: redact-data
  {{- if .Values.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ .Values.persistence.claimName | quote }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- with .Values.extraVolumes }}
{{ toYaml . }}
{{- end }}
{{- end -}}
