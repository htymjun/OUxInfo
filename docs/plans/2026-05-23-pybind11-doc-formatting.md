# Plan: Fix pybind11 API Documentation Formatting

## Context

The OUxInfo project uses **MkDocs** (not Sphinx) with the `mkdocstrings[python]` plugin for API docs deployed to GitHub Pages. The spec references Sphinx/Napoleon conventions, but the same underlying fixes apply — MkDocs with `docstring_style: numpy` is the equivalent.

Two root causes for the rendering issues:
1. pybind11 prepends an auto-generated function signature (with `typing.Annotated[...]`) to each function's `__doc__`, which mkdocstrings then displays.
2. The `Returns` sections in all 8 C++ docstrings use the unnamed format (`float\n    description`) instead of the named format required by NumPy style (`value : float\n    description`).

A third unrelated breakage: `mkdocs.yml` references `docs/theory.md` in `nav`, but that file was deleted (shown in git status as `D docs/theory.md`), causing an mkdocs build error.

---

## Changes

### File 1: `ouxinfo/ouxinfo.cpp`

**Change A — Disable auto-generated signatures (lines ~509)**

Add a `py::options` RAII object immediately after `PYBIND11_MODULE(_core, m) {` and before the first `m.def()`. This applies to all 8 bindings in scope.

```cpp
PYBIND11_MODULE(_core, m) {
    m.doc() = "...";

    py::options options;
    options.disable_function_signatures();

    m.def("shannon_entropy", ...);
    // ...all other m.def calls unchanged...
}
```

> Note: the spec shows `py::options::disable_function_signatures()` passed inside `m.def()`; that is not valid pybind11 syntax. The RAII pattern above is correct and covers all 8 functions with one addition.

**Change B — Fix `Returns` sections (all 8 docstrings)**

Every `Returns` block uses unnamed format. Add a variable name before the type on the return-type line:

| Function | Before | After |
|---|---|---|
| `shannon_entropy` | `float` | `value : float` |
| `KL_div` | `float` | `value : float` |
| `mutual_info` | `float` | `value : float` |
| `conditional_mutual_info` | `float` | `value : float` |
| `transfer_entropy` | `float` | `value : float` |
| `information_flow` | `float` | `value : float` |
| `transfer_entropy_causal_map` | `ndarray of shape (N, N)` | `value : ndarray of shape (N, N)` |
| `information_flow_causal_map` | `tuple of three ndarray...` | `value : tuple of three ndarray...` |

**Change C — Fix `information_flow` Notes equation (line ~656)**

Move the inline equation to an indented block:

```
Before:  Based on the decomposition dI/dt = dI_X + dI_Y + Leak.

After:   Based on the decomposition

             dI/dt = dI_X + dI_Y + Leak
```

---

### File 2: `mkdocs.yml`

**Change A — Remove deleted `theory.md` from nav**

```yaml
nav:
  - Home: index.md
  - Installation: install.md
  - API Reference: api.md   # remove "Theory: theory.md"
```

**Change B — Disable signature annotations**

```yaml
options:
  show_signature_annotations: false   # was: true
```

This ensures that even if mkdocstrings finds any residual annotation metadata, it won't render `typing.Annotated[...]` in the output.

---

## Critical Files

- [`ouxinfo/ouxinfo.cpp`](ouxinfo/ouxinfo.cpp) — lines 509–718 (PYBIND11_MODULE block)
- [`mkdocs.yml`](mkdocs.yml) — nav section and mkdocstrings options

---

## Verification

1. Build the package: `./rebuild.sh`
2. Build and serve docs: `mkdocs serve`
3. Open `http://127.0.0.1:8000/api/` and verify:
   - Function signatures show as `ouxinfo.information_flow()` (no `typing.Annotated`)
   - `Parameters` and `Returns` render as structured tables
   - `Returns` entries show names (e.g., `value`)
   - The Notes equation for `information_flow` is on its own indented line
   - No 404 for Theory page

> After PR merge, also save a copy of this plan to `docs/plans/2026-05-23-pybind11-doc-formatting.md` per CLAUDE.md convention.
