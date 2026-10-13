# Releasing llm-redact

Releases are tag-driven: pushing a `v*` tag runs `.github/workflows/release.yml`,
which builds and verifies the distribution, creates a GitHub Release with the
sdist, the wheel and two SBOMs attached, publishes `llm-redact-proxy` to PyPI,
and publishes the stock and `-ner` container images to GHCR. Creating the tag
is the release: PyPI has the files within a few minutes of the tag, and a
version published there cannot be replaced.

## Preflight checklist

- CI is green on the exact commit you are about to tag. Tags do not trigger CI
  (`ci.yml` runs on pushes to `main`, on pull requests and weekly), so the
  push run that follows the merge to `main` is the evidence. It takes about 70
  to 125 minutes; the `ner models` and `mutation testing` jobs are the long
  ones. If a job fails, re-run it from the Actions page (a re-run keeps the
  commit) rather than tagging past it.
- `main` stays frozen from that merge until the tag is pushed. Any later push
  to `main` cancels the CI run above (the workflow's concurrency group is its
  ref, with cancel-in-progress), and so does the weekly scheduled run, which
  runs on `main` too: merge nothing else, the Homebrew PR of the previous
  release included. If the weekly run cancels the push run, the weekly run,
  on the same commit, is the evidence. Until the tag run has pushed the
  images, `main` names images that do not exist yet (the Helm chart's
  default tag, the sidecar manifest), which is one more reason to keep the
  window short.
- `CHANGELOG.md`: the `[Unreleased]` section has been moved into a dated
  `## [X.Y.Z] - YYYY-MM-DD` section describing exactly that tree.
- `src/llm_redact/__init__.py` `__version__` matches `X.Y.Z` (the single
  source of truth; hatchling reads it at build time). The workflow's first
  step after the checkout enforces this: a tag that does not match
  `__version__` fails the `build` job before anything is built or published —
  so a tag pushed before the version-bump PR lands on `main` fails fast
  instead of releasing artifacts labeled with the previous version.
- The release-prep commit changes every place that carries the version, not
  only `__version__`: `deploy/helm/llm-redact/Chart.yaml` (`version` and
  `appVersion`), `plugins/llm-redact/.claude-plugin/plugin.json` (`version`)
  and the image tag in `deploy/k8s-sidecar.yaml`. Each is pinned to
  `__version__` by a test (`tests/test_deploy_assets.py`,
  `tests/test_plugins.py`), so a missed one fails CI.

## Cutting the release

Two ways start the same run, because the run is triggered by the tag being
pushed. Either way, name the commit by its full SHA. A branch name such as
`main` resolves to whatever `main` is at that moment (the recent releases
used it, which is right only while `main` is frozen). With `main` frozen,
the commit is its tip:

```bash
git fetch origin main
sha=$(git rev-parse origin/main)
git show "$sha":src/llm_redact/__init__.py | grep '^__version__'   # must show X.Y.Z
```

From git, make the tag annotated: it records tagger, date, and message, and
`git describe` considers only annotated tags by default. The workflow then
creates the GitHub Release itself, with generated notes.

```bash
git tag -a vX.Y.Z -m "llm-redact X.Y.Z" "$sha"
git push origin vX.Y.Z
```

Or publish a GitHub Release (the Releases UI or the API) whose tag target is
that same commit. GitHub creates the tag, a lightweight one, when the Release
is published; the workflow then attaches its files to that Release instead of
creating a new one, and never edits its notes, so write them when you publish
(the UI can generate them; the API takes `generate_release_notes`):

```bash
gh api -X POST repos/asanderson/llm-redact/releases \
  -f tag_name=vX.Y.Z -f target_commitish="$sha" -f name=vX.Y.Z \
  -F generate_release_notes=true
```

Nothing in the workflow depends on the kind of tag: it uses the tag's name
and the commit the tag points to.

## What the workflow produces

- **GitHub Release** `vX.Y.Z` with four files: the sdist
  `llm_redact_proxy-X.Y.Z.tar.gz`, the wheel
  `llm_redact_proxy-X.Y.Z-py3-none-any.whl`, and two CycloneDX SBOMs,
  `llm-redact-runtime.cdx.json` (the runtime closure) and
  `llm-redact-ner-image.cdx.json` (the `-ner` image's closure). The notes are
  auto-generated when the workflow creates the Release; a Release you
  published first keeps the notes you gave it.
- **GHCR images**, each for `linux/amd64` and `linux/arm64`: the stock image
  `ghcr.io/asanderson/llm-redact` tagged `X.Y.Z`, `X.Y` and `latest`, and the
  `-ner` variant (the `hf` and `gliner` NER extras on a CPU-only torch) tagged
  `X.Y.Z-ner`, `X.Y-ner` and `latest-ner`. One `publish-ghcr` job each, so a
  failed `-ner` build never holds back the stock image. Both carry a BuildKit
  SBOM attestation, and each image's multi-platform index (the digest its
  tags point to) is signed with cosign (keyless); the per-platform images
  listed in it carry no signature of their own.
- **PyPI release** `llm-redact-proxy X.Y.Z` via trusted publishing (OIDC, no
  stored secrets). The repository is public, so `publish-pypi` runs on every
  tag (it skips itself on a private repository). It needs only `build`, so it
  starts as soon as that job finishes, and PyPI has the files within a few
  minutes of the tag, independent of the image jobs and of CI. It builds the
  distribution again from the tagged
  commit. **A published version cannot be replaced**: PyPI never lets a file
  name be reused, even after a deletion, so a mistake that reaches PyPI is
  fixed only by the next patch version. That covers the sdist's contents too:
  `pyproject.toml` configures only the wheel, so the sdist is the whole
  repository tree as checked out (`CHANGELOG.md`, `CLAUDE.md`, `docs/`,
  `tests/`, `.github/` ...), and a mistake in any of those files at the tagged
  commit is as permanent as the version.
- **Build provenance attestations** (Sigstore) for the four files attached
  to the GitHub Release — on PUBLIC repos only: GitHub cannot persist
  attestations for user-owned private repos, so the step self-skips there
  (a pre-debut release run failed on exactly this and was withdrawn). PyPI uploads additionally carry PEP 740
  attestations emitted by the trusted publisher — nothing extra to do.

The CI `container-ner` job builds the `-ner` variant in every CI run, but on
`linux/amd64` only. Its `linux/arm64` half is built under QEMU emulation, and
only by the tag run: nothing before the tag exercises it. To rehearse it
first, build it by hand on an arm64 host or under QEMU:

```bash
docker buildx build --platform linux/arm64 --target runtime \
  --build-arg 'NER_EXTRAS=--extra hf --extra gliner' .
```

Verify: the Releases page shows the four files; each `publish-ghcr` job log
shows the image's tags (the metadata step's `Docker tags` group) and its
pushed digest (the build step's `Digest` group, the digest the sign step
signs); `docker buildx imagetools inspect ghcr.io/asanderson/llm-redact:X.Y.Z-ner`
prints that digest on its top-level `Digest:` line, above both platforms;
the attestation checks out against a downloaded artifact; and each image's
signature verifies:

```bash
gh release download vX.Y.Z --repo asanderson/llm-redact --dir /tmp/rel
gh attestation verify /tmp/rel/llm_redact_proxy-X.Y.Z-py3-none-any.whl \
  --repo asanderson/llm-redact

docker buildx imagetools inspect ghcr.io/asanderson/llm-redact:X.Y.Z | grep -m1 '^Digest:'   # the <digest> below
cosign verify ghcr.io/asanderson/llm-redact@<digest> \
  --certificate-identity-regexp '^https://github\.com/asanderson/llm-redact/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Run the `cosign verify` once for the stock image's digest and once for the
`-ner` image's (`X.Y.Z-ner`), each the top-level `Digest:` line, the
multi-platform index: only that digest is signed, and cosign finds no
signature on a per-platform digest. The signature is a Sigstore bundle
attached to the index (an OCI referrer; there is no `.sig` tag), so it takes
a cosign release that verifies Sigstore bundle signatures: cosign 3, which
the workflow signs with, verifies it as shown; cosign 2.6 needs
`--new-bundle-format` added. The signing certificate's identity is this
workflow at the release tag, and the regexp pins exactly that. cosign
matches a regexp anywhere in the identity (so the `^` and `$` matter), and
a looser pin such as `https://github.com/asanderson/llm-redact/.*` also
accepts the CI workflow or a branch.

After PyPI lists the new version, point the Homebrew formula at the new
sdist and land it via a normal PR. The script reads PyPI's JSON for that
version, so it needs network access and fails (HTTP 404, formula untouched)
until the version is listed:

```bash
python scripts/update_formula.py X.Y.Z
git checkout -b formula-X.Y.Z && git commit -am "Point brew formula at X.Y.Z" && git push -u origin formula-X.Y.Z
```

(The formula's dependency resources track `uv.lock` — refresh their
pins in the same PR whenever the lock's runtime closure changed.) Merge it
only once the tagged commit's CI run has finished: the merge is a push to
`main`, which cancels a run still in progress.

## One-time user setup

Both steps are done for `asanderson/llm-redact`: `ghcr.io/asanderson/llm-redact`
pulls anonymously, and PyPI lists `llm-redact-proxy` releases from 1.0.0 on.
They apply again to a fork, or to a new image or project name.

- **GHCR visibility**: packages published from Actions with `GITHUB_TOKEN`
  are created *private by default*, even on public repos. For anonymous
  `docker pull`, flip `ghcr.io/asanderson/llm-redact` to public once:
  package page → Package settings → Change visibility.
- **PyPI trusted publishing** (do once, *before* tagging the first
  PyPI-published release — the `publish-pypi` job runs on public repos
  only, and fails there without this setup):
  1. On pypi.org → Your account → Publishing, add a **pending publisher**
     for the project name `llm-redact-proxy`: repository
     `asanderson/llm-redact`, workflow `release.yml`, environment `pypi`.
     (A pending publisher is for a project name PyPI does not have yet; on
     an existing project, add a trusted publisher in its settings instead.)
  2. Create a GitHub environment named `pypi` in the repo settings
     (Settings → Environments → New environment).

  Publishing to PyPI makes the sdist/wheel — and therefore the source —
  public, independent of the repository's visibility.

## If the first run of a release fails

Tag-triggered workflows execute the workflow file **as of the tagged
commit** — fixing `release.yml` on `main` afterwards does not retro-apply to
an existing tag, and a re-run of the run uses the tagged commit too. What
is left to recover depends on which job failed: `publish-pypi` and the two
`publish-ghcr` jobs all start when `build` finishes and do not depend on
each other.

| Failed | What exists afterwards | Recovery |
| --- | --- | --- |
| `build`, the version guard | Nothing: the other jobs need `build` and are skipped. A Release made through the UI or API stays, empty. | Delete the tag and that Release, land the fix on `main`, tag the new tip. |
| `build`, a step between the guard and the Release step (build, `twine check`, an SBOM, the attestation or its check, the artifact upload) | Nothing on PyPI or GHCR (the other jobs are skipped), and no Release unless you made one through the UI or API. | A transient fault (runner, network, Sigstore): "Re-run failed jobs". Anything else: as for the guard. |
| `build`, the Release step (the last one) | Nothing on PyPI or GHCR (the other jobs are skipped). A Release may exist, with some of its files. | "Re-run failed jobs": an existing Release gets the files uploaded to it with `--clobber`. A cause in the workflow file: as for the guard. |
| `publish-pypi`, before PyPI accepted a file (the `pypi` environment, the trusted-publisher setting, an outage) | The Release and the images are unaffected; PyPI has nothing. | Fix the setting, then "Re-run failed jobs". The workflow file is unchanged, so no new tag is needed. |
| `publish-pypi`, after PyPI accepted a file | The version is on PyPI for good. | Never re-tag; the next patch version carries any fix. |
| One `publish-ghcr` job (`stock` or `ner`) | The other image, PyPI and the Release are unaffected (`fail-fast` is off). If the cosign step failed, the tags are pushed but unsigned. | A transient fault: "Re-run failed jobs" re-runs that job alone (it uses no artifacts): it rebuilds, pushes and signs again. A fault in the workflow or the Dockerfile cannot be fixed under this tag, and `publish-pypi` has normally finished by then: ship the fix as the next patch version. |

Deleting the tag and re-tagging is acceptable only while nothing has
consumed the artifacts. Check first — the PyPI listing, the Release's download
counts, and whether the image tags exist:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/llm-redact-proxy/X.Y.Z/json   # 404: not on PyPI
gh api repos/asanderson/llm-redact/releases/tags/vX.Y.Z --jq '.assets[] | [.name, .download_count] | @tsv'
docker buildx imagetools inspect ghcr.io/asanderson/llm-redact:X.Y.Z | head -n 3       # "not found": no stock image
docker buildx imagetools inspect ghcr.io/asanderson/llm-redact:X.Y.Z-ner | head -n 3   # "not found": no -ner image
```

Then:

```bash
git push --delete origin vX.Y.Z        # remove the remote tag
# delete the partial GitHub Release, if one was created
# land the workflow fix on main via a normal PR, then re-tag the new tip
```

Never re-tag a version that has been announced or published to PyPI —
publish a patch release instead.
