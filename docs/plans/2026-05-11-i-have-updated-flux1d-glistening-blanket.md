# Plan: Save all test results to ./tests/results/

## Context

The user wants all test visualizations saved under `./tests/results/`. This covers:
1. Fixing `tutorials/test_flux1d.py` to save to `tests/results/` instead of `tutorials/data/`
2. Adding `if __name__ == '__main__':` plot-and-save blocks to all five test files in `tests/` (H, IF, KL, MI, TE)
3. Updating `tutorials/test_TE.py` and `tutorials/test_IF.py` to `savefig` instead of `show`

All plots use `plt.rcParams` from `ouxinfo.utils.myParams` style (Times New Roman, STIX math, inward ticks, fontsize 20), `dpi=150, bbox_inches='tight'`.

---

## Files to Modify

| File | Change |
|------|--------|
| [tutorials/test_flux1d.py](tutorials/test_flux1d.py) | Change save path `tutorials/data/` → `tests/results/`; update `os.makedirs` path |
| [tutorials/test_TE.py](tutorials/test_TE.py) | Replace `plt.show()` → `savefig('tests/results/TE_coupling.png')` + `close()`; add `os.makedirs` |
| [tutorials/test_IF.py](tutorials/test_IF.py) | Replace `plt.show()` → `savefig('tests/results/IF_coupling.png')` + `close()`; add `os.makedirs` |
| [tests/test_H.py](tests/test_H.py) | Add `if __name__ == '__main__':` block → `tests/results/H.png` |
| [tests/test_KL.py](tests/test_KL.py) | Add `if __name__ == '__main__':` block → `tests/results/KL.png` |
| [tests/test_MI.py](tests/test_MI.py) | Add `if __name__ == '__main__':` block → `tests/results/MI.png` |
| [tests/test_IF.py](tests/test_IF.py) | Add `if __name__ == '__main__':` block → `tests/results/IF.png` |
| [tests/test_TE.py](tests/test_TE.py) | Add `if __name__ == '__main__':` block → `tests/results/TE.png` |

---

## Plot Details

### tutorials/test_flux1d.py
Change 3 savefig paths: `tutorials/data/flux1d_*.png` → `tests/results/flux1d_*.png`

### tutorials/test_TE.py
Replace `plt.show()` with:
```python
plt.savefig('tests/results/TE_coupling.png', dpi=150, bbox_inches='tight')
plt.close()
```
Also add `import os` and `os.makedirs('tests/results', exist_ok=True)` after the imports.

### tutorials/test_IF.py
Same pattern as test_TE.py → `tests/results/IF_coupling.png`

### tests/test_H.py → `tests/results/H.png`
Two-panel figure reusing existing logic:
- Left: H_est (circles) vs H_true (line) over σ ∈ [1, 5], k=5
- Right: H_est (circles) vs H_true (hline) over k ∈ [3, 19], σ=1

### tests/test_KL.py → `tests/results/KL.png`
Two-panel figure:
- Left: KL_est (circles) vs KL_true (line) over σ_y ∈ [2, 3, 4, 5], k=5
- Right: KL_est (circles) vs KL_true (hline) over k ∈ [3, 19], σ_x=1, σ_y=3

### tests/test_MI.py → `tests/results/MI.png`
Two-panel figure:
- Left: MI_est (circles) vs MI_true (line) over ρ ∈ [0.6, 0.9], k=5
- Right: MI_est (circles) vs MI_true (hline) over k ∈ [3, 14]

### tests/test_IF.py → `tests/results/IF.png`
Single figure reusing `_gaussian_ar_system` and `_analytical_if_gaussian_ar`:
- IF_est(x→y) (circles) and IF_true (line) vs c_xy ∈ [0.3, 0.5, 0.7], a=b=0.5, k=5, N=5000
- IF_est(y→x) (triangles) for comparison

### tests/test_TE.py → `tests/results/TE.png`
Single figure reusing `_gaussian_ar_system` and `_analytical_te_gaussian_ar`:
- TE_est(x→y) (circles) and TE_true (line) vs c_xy ∈ [0.3, 0.5, 0.7], a=b=0.5, k=5, N=5000

---

## Common matplotlib settings (applied to all new plots)
```python
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 20
```

---

## Verification

```bash
# Change flux1d save path and run tutorial
python tutorials/test_flux1d.py

# Run all test __main__ blocks
python tests/test_H.py
python tests/test_KL.py
python tests/test_MI.py
python tests/test_IF.py
python tests/test_TE.py

# Run tutorial scripts
python tutorials/test_TE.py
python tutorials/test_IF.py

# Confirm all files saved
ls tests/results/

# Confirm pytest still passes
pytest tests/ -v --ignore=tests/test_speed.py
```

Expected files in `tests/results/`:
`flux1d_map.png`, `flux1d_J.png`, `flux1d_Jsym.png`,
`H.png`, `KL.png`, `MI.png`, `IF.png`, `TE.png`,
`TE_coupling.png`, `IF_coupling.png`
