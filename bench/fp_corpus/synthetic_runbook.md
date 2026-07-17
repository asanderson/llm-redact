# On-call runbook: token rotation (SPECIMENS ONLY — every value below is
# structurally invalid on purpose; rotating them does nothing)

## Vendor token shapes we rotate, with deliberately-broken examples

Tailscale keys look like `tskey-api-kXXXXXXXXXX-...`; a bare `tskey` or a
`tskey-` with a short tail like tskey-api-k12 is a placeholder, not a key.

DigitalOcean tokens are dop_v1_ plus 64 hex. The doc placeholder
dop_v1_0123456789abcdef is too short to be real, and dop_v2_ does not
exist as a prefix family.

Notion legacy tokens are secret_ plus exactly 43 alphanumerics; the
secretsmanager path secret_prod_backend_v2 and the field secret_value are
ordinary identifiers. New-style placeholders are written ntn_XXXXXXXX in
our docs.

Linear: lin_api_ plus the body. lin_api_KEYGOESHERE is the template
string in the wiki (too short to match anything real).

Supabase: sbp_ plus 40 hex; sbp_deadbeef is the sandbox placeholder.
Publishable keys (sb_publishable_...) are not secrets and are committed
in the frontend repo on purpose: sb_publishable_a1B2c3D4e5F6g7H8i9J0k1.

PlanetScale passwords look like pscale_pw_...; the CI variable is named
PSCALE_PW_STAGING and the helm value is pscale_pw_placeholder.

Doppler: dp.pt. personal tokens. The literal dp.pt.example in docs, the
filename dp.pt.bak, and the version string ddp.pt.4.1 stay quiet.

Postman: PMAK- plus 24+34 hex. PMAK-REDACTED and
PMAK-0123456789abcdef-0123 are the runbook's own scrubbed forms.

Airtable: pat + 14 chars + dot + 64 hex. Ordinary words containing pat —
path, pattern, dispatch, patch-2 — and the short patAbCdEfGh123.token
form in older docs never match the real grammar.

Shopify admin tokens are shpat_ plus 32 hex; shpat_EXAMPLE and the
16-hex shpat_0123456789abcdef doc form are non-tokens.

## Rotation steps
1. Page the owning team in #oncall-infra (see rota sheet row 42).
2. Generate the replacement in the vendor console.
3. Update the Doppler config: doppler secrets set --silent
4. Restart consumers: kubectl rollout restart deploy/api-gateway
5. Verify burn-down: the old token must 401 within 15 minutes.

Escalation: if any consumer still authenticates with the revoked value
after 30 minutes, treat as an incident (SEV-2 template, incident-9917).
