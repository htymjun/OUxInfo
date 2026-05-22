# GitHub Pages for OUxInfo

**Date:** 2026-05-22

## Goal

Publish an OUxInfo documentation site via GitHub Pages with:
- Auto-generated API docs from source (pybind11 docstrings + Python docstrings)
- Installation guide
- Mathematical theory background

## Approach

**MkDocs + Material theme + mkdocstrings[python]**

- MkDocs fits naturally: existing docs are Markdown
- Material theme provides professional look with dark mode and math rendering (MathJax)
- mkdocstrings[python] (griffe-based) generates API docs by introspecting the live Python package
- GitHub Actions builds the C++ extension first so `ouxinfo._core` is importable at docs-build time

## Files Created / Modified

| File | Action |
|---|---|
| `mkdocs.yml` | New — MkDocs config with Material, mkdocstrings, MathJax |
| `docs/index.md` | New — home page with badges, features, quick start, citation |
| `docs/api.md` | Rewritten — mkdocstrings `:::` directives for all public functions |
| `docs/javascripts/mathjax.js` | New — MathJax config for Material theme |
| `docs/css/extra.css` | New — minor style tweaks |
| `.github/workflows/docs.yml` | New — build & deploy on push to main |
| `ouxinfo/ouxinfo.cpp` | Modified — pybind11 docstrings expanded to full numpy-style |

## Key Design Decisions

1. **C++ docstrings in source**: Expanded pybind11 one-liners in `ouxinfo.cpp` to full numpy-style
   raw string literals `R"doc(...)doc"`. This makes C++ function docs live in the source code,
   satisfying the "auto-generated from source" requirement.

2. **Build C++ in CI**: The docs workflow installs GCC and builds the extension with
   `pip install -e .` before running mkdocs. This lets mkdocstrings import `ouxinfo._core`
   at build time without needing separate stub files.

3. **GitHub Pages deployment**: `mkdocs gh-deploy --force` pushes the built site to the
   `gh-pages` branch. Enable GitHub Pages in repo Settings → Pages → Source: `gh-pages` branch.

## Verification

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
pip install -e .   # build C++ extension
mkdocs serve       # preview at http://127.0.0.1:8000
```
