# Releasing llm-redact

Releases are tag-driven: pushing a `v*` tag runs `.github/workflows/release.yml`,
which builds and verifies the distribution, creates a GitHub Release with the
sdist and wheel attached, and publishes the container image to GHCR.

## Preflight checklist

- CI is green on the `main` commit you are about to tag.
- `CHANGELOG.md`: the `[Unreleased]` section has been moved into a dated
  `## [X.Y.Z] - YYYY-MM-DD` section describing exactly that tree.
- `src/llm_redact/__init__.py` `__version__` matches `X.Y.Z` (the single
  source of truth; hatchling reads it at build time).

## Cutting the release

```bash
git fetch origin main
git tag -a vX.Y.Z -m "llm-redact X.Y.Z" <main-tip-sha>
git push origin vX.Y.Z
```

Use an annotated tag: it records tagger, date, and message, and is what
`git describe` and release tooling treat as a release object.

Creating the tag through GitHub's Releases UI also works: that flow creates
the release object up front, and the workflow attaches the built dist files
to it instead of creating a new one (auto-generated notes only happen on the
CLI-tag path, so write the release notes yourself in the UI).

## What the workflow produces

- **GitHub Release** `vX.Y.Z` with `llm_redact_proxy-X.Y.Z.tar.gz` and
  `llm_redact_proxy-X.Y.Z-py3-none-any.whl` attached and auto-generated notes.
- **GHCR image** `ghcr.io/asanderson/llm-redact:X.Y.Z` and `:X.Y`.
- **PyPI release** `llm-redact-proxy X.Y.Z` via trusted publishing (OIDC, no
  stored secrets) — **held while the repository is private**: the
  `publish-pypi` job self-skips on private repos, because publishing the
  sdist/wheel would make the source public regardless of repo visibility.
  Making the repo public is the deliberate switch that arms it; the job
  then still fails until the one-time setup below is done.
- **Build provenance attestations** (Sigstore) for the dist files attached
  to the GitHub Release — on PUBLIC repos only: GitHub cannot persist
  attestations for user-owned private repos, so the step self-skips there
  (this is why the v0.6.0 tag's run failed and that release is held). PyPI uploads additionally carry PEP 740
  attestations emitted by the trusted publisher — nothing extra to do.

Verify: the Releases page shows both dist files; the `publish-ghcr` job log
shows the pushed digest and tags; and the attestation checks out against
the downloaded artifact:

```bash
gh release download vX.Y.Z --repo asanderson/llm-redact --dir /tmp/rel
gh attestation verify /tmp/rel/llm_redact_proxy-X.Y.Z-py3-none-any.whl \
  --repo asanderson/llm-redact
```

(GHCR image attestation is a documented follow-up; the image is not signed
yet.)

After the PyPI publish succeeds, point the Homebrew formula at the new
sdist and land it via a normal PR:

```bash
python scripts/update_formula.py X.Y.Z
git checkout -b formula-X.Y.Z && git commit -am "Point brew formula at X.Y.Z" && git push
```

(The formula's dependency resources track `uv.lock` — refresh their
pins in the same PR whenever the lock's runtime closure changed.)

## One-time user setup

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
  2. Create a GitHub environment named `pypi` in the repo settings
     (Settings → Environments → New environment).

  Publishing to PyPI makes the sdist/wheel — and therefore the source —
  public, independent of the repository's visibility.

## If the first run of a release fails

Tag-triggered workflows execute the workflow file **as of the tagged
commit** — fixing `release.yml` on `main` afterwards does not retro-apply to
an existing tag. Recovery, acceptable only while nothing has consumed the
artifacts:

```bash
git push --delete origin vX.Y.Z        # remove the remote tag
# delete the partial GitHub Release, if one was created
# land the workflow fix on main via a normal PR, then re-tag the new tip
```

Never re-tag a version that has been announced or published to PyPI —
publish a patch release instead.
