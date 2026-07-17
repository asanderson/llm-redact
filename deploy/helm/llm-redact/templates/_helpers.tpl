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
{{- if eq .Values.mode "standalone" -}}
{{- $multi := or .Values.autoscaling.enabled (gt (int .Values.replicaCount) 1) -}}
{{- if and $multi (or (eq .Values.vault.backend "memory") (eq .Values.vault.backend "sqlite")) -}}
{{- fail "llm-redact: a standalone autoscaled/multi-replica proxy needs a SHARED vault (vault.backend must be postgresql/mysql/oracle/dbapi) — a per-pod memory/sqlite vault would issue inconsistent tokens across replicas (never-wrong-value). Set vault.backend to a server backend, or set autoscaling.enabled=false and replicaCount=1." -}}
{{- end -}}
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
