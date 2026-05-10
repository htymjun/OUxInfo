# Plan: Tutorial for 1D Information Flux

## Context

A simple tutorial script demonstrating `information_flux_1d` is needed in `tutorials/`. It should match the existing tutorial style (no pytest, matplotlib visualization, `plt.rcParams` block, `.e0` float notation) and show the directional flux quantities on a synthetic 1D driven AR chain.

---

## File to Create

`tutorials/test_flux1d.py`

---

## Style Conventions (from existing tutorials)

- `plt.rcParams` block (same 5 lines used in test_TE.py, test_EE.py, test_shan.py)
- `plt.figure(figsize=(7,7))` or `(6,6)` for line plots
- `plt.xlabel/ylabel(fontsize=20, style='italic')`
- `plt.legend(frameon=False)` when used
- `plt.show()` — no `savefig`, no `tight_layout`
- Colors: `'blue'`, `'red'`, `'black'`; linestyles: `'solid'`, `'dashed'`
- Float notation: `0.5e0`, `1.0e0`, etc.
- No `myParams`, no pytest imports

---

## Data: 1D Driven AR Chain

Unidirectional coupling left → right (exact same pattern as `tests/test_flux1d.py`):

```
data[0, t+1] = alpha * data[0, t] + noise
data[i, t+1] = alpha * data[i, t] + beta * data[i-1, t] + noise  (i > 0)
```

Parameters: `nx=6`, `nt=3000`, `alpha=0.5e0`, `beta=0.5e0`, `sigma=1.0e0`, seed=42.  
Interface positions: `x_iface = np.arange(nx - 1) + 0.5e0` (half-integer indices matching the spec).

---

## Three Plots

**Plot 1 — Spatiotemporal map** (imshow, first 200 time steps)  
- `plt.figure(figsize=(8, 4))`, `cmap='RdBu_r'`, `aspect='auto'`, `origin='lower'`  
- x-axis: `t`, y-axis: `i`, colorbar labeled `q_i(t)`

**Plot 2 — Directional fluxes vs interface** (line plot)  
- `plt.figure(figsize=(7, 7))`
- Three lines: `J_fwd` (blue solid), `J_bwd` (red solid), `J_net` (black dashed)
- `axhline(0)` in gray dotted
- x-axis: `i + 1/2`, y-axis: `information flux`
- `plt.legend(frameon=False)` with LaTeX labels

**Plot 3 — Coupling strength vs leak** (line plot)  
- `plt.figure(figsize=(7, 7))`
- Two lines: `J_sym` (blue solid), `Leak_fwd` (red dashed)
- x-axis: `i + 1/2`, y-axis: `information flux`
- `plt.legend(frameon=False)` with LaTeX labels

---

## Verification

```bash
cd /home/jhatayama/ShannonEntropyEstimator/tutorials
python test_flux1d.py
```

Expected: three matplotlib windows open sequentially; J_net > 0 at each interface; Leak smaller than J_sym.
