# Plan: Single-source version with automatic GitHub tagging

## Context

The version `v0.1.4` is currently a hardcoded string in two places (`setup.py` and `conda-recipe/ouxinfo/meta.yaml`) with no GitHub Actions workflow that creates the corresponding git tag. When the user bumps the version, they must edit multiple files and manually run `git tag` — the parts are out of sync by design.

The fix has two parts:
1. Make **one file** the single source of truth for the version string, with all other locations reading from it.
2. Add a **GitHub Actions workflow** that automatically creates a git tag (and GitHub Release) whenever the version in that file changes on `main`.

---

## Changes

### New file: `ouxinfo/_version.py`

```python
__version__ = "0.1.4"
```

This is the only place the version string lives.

---

### `setup.py`

Replace the hardcoded `version="0.1.4"` with a dynamic read:

```python
# near top of setup.py, before setup()
import re, pathlib

def _get_version():
    text = pathlib.Path("ouxinfo/_version.py").read_text()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text).group(1)

setup(
    name="ouxinfo",
    version=_get_version(),
    ...
)
```

---

### `ouxinfo/__init__.py`

Add one import so `ouxinfo.__version__` works at runtime:

```python
from ouxinfo._version import __version__
```

---

### `conda-recipe/ouxinfo/meta.yaml`

This file **still requires a manual edit** — it also needs a SHA256 that can only be computed after uploading to PyPI, so it cannot be automated. The plan is to leave it as-is and document that it is updated after every PyPI release, not before.

---

### New file: `.github/workflows/tag-release.yml`

Triggers on push to `main`. Reads the version from `_version.py`, checks whether a tag `v{version}` already exists, and creates the tag (and a draft GitHub Release) if it does not.

```yaml
name: Tag and release on version bump

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  tag:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # need full history to check existing tags

      - name: Read version
        id: version
        run: |
          VERSION=$(python -c "
          import re, pathlib
          text = pathlib.Path('ouxinfo/_version.py').read_text()
          print(re.search(r'__version__\s*=\s*[\"\\']([^\"\\']+)', text).group(1))
          ")
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"

      - name: Check if tag exists
        id: tag_check
        run: |
          TAG="v${{ steps.version.outputs.version }}"
          if git rev-parse "$TAG" >/dev/null 2>&1; then
            echo "exists=true" >> "$GITHUB_OUTPUT"
          else
            echo "exists=false" >> "$GITHUB_OUTPUT"
          fi

      - name: Create tag and release
        if: steps.tag_check.outputs.exists == 'false'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          TAG="v${{ steps.version.outputs.version }}"
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git tag "$TAG"
          git push origin "$TAG"
          gh release create "$TAG" \
            --title "$TAG" \
            --generate-notes \
            --draft
```

The release is created as a **draft** so the user can review and publish it manually with any additional release notes.

---

## After this change — bumping the version

1. Edit `ouxinfo/_version.py`: `__version__ = "0.1.5"`
2. Commit and push to `main`
3. GitHub Actions reads the version, sees no tag `v0.1.5`, creates it automatically and opens a draft release
4. After PyPI upload, manually update `conda-recipe/ouxinfo/meta.yaml` with the new version and SHA256

That is **one code edit** instead of three.

---

## Critical files

- [`ouxinfo/_version.py`](ouxinfo/_version.py) — new, single source of truth
- [`setup.py`](setup.py) — remove hardcoded version, read from `_version.py`
- [`ouxinfo/__init__.py`](ouxinfo/__init__.py) — expose `__version__`
- [`.github/workflows/tag-release.yml`](.github/workflows/tag-release.yml) — new workflow

---

## Verification

1. `python -c "import ouxinfo; print(ouxinfo.__version__)"` → prints `0.1.4`
2. `python setup.py --version` → prints `0.1.4`
3. Push to main → GitHub Actions creates tag `v0.1.4` (or skips if it already exists)
4. Change to `0.1.5`, push → tag `v0.1.5` is created, draft release appears in GitHub
