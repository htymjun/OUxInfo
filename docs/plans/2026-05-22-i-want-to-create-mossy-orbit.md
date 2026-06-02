# Plan: Exclude docs/plans/ and fix GitHub Pages source

**Date:** 2026-05-22

## Context

The live GitHub Pages site currently shows README.md because GitHub Pages source is still set to "Deploy from a branch" (root). The MkDocs workflow is in place but the repo Settings haven't been switched to "GitHub Actions" yet.

Additionally, `docs/plans/` is being included in the built MkDocs site — the files are compiled to HTML even though they don't appear in the nav, generating warnings and leaking internal plan files publicly.

## Changes

### 1. `mkdocs.yml` — add `exclude_docs`

Add one line so MkDocs completely ignores `docs/plans/` during the build:

```yaml
exclude_docs: |
  plans/
```

This uses MkDocs 1.5+ `exclude_docs` config key which prevents matched files from being copied or rendered into `site/`.

### 2. Manual GitHub step (not a code change)

In the repository **Settings → Pages → Build and deployment → Source**, change from  
`Deploy from a branch` → `GitHub Actions`.

This tells GitHub to use the artifact produced by `jekyll-gh-pages.yml` instead of running Jekyll on the root.

## File to Modify

| File | Change |
|---|---|
| `mkdocs.yml` | Add `exclude_docs: plans/` block |

## Verification

```bash
source ~/htymenv_new/bin/activate
mkdocs build
# Confirm site/plans/ does NOT exist:
ls site/ | grep plans   # should print nothing
```
