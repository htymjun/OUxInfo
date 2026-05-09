# Plan: GitHub Actions CI for pytest

## Context

The repo now has a pytest suite (`tests/`) but no CI. The goal is to run `pytest tests/` automatically on every push and pull request. `ouxinfo` is published on PyPI (binary wheel), so no local build step is needed; gcc/g++ must be installed explicitly to provide the OpenMP runtime (`libgomp`) that the wheel links against.

## Files to Create

### 1. `docs/plans/2026-05-09-github-actions-pytest.md`
Project-side plan doc required by the repo convention.

### 2. `.github/workflows/test.yml`

```yaml
name: Test

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install GCC and G++
        run: sudo apt-get update && sudo apt-get install -y gcc g++

      - name: Install dependencies
        run: pip install ouxinfo pytest

      - name: Run tests
        run: pytest tests/ -v
```

**Key decisions:**
- `python-version: "3.12"` — minimum supported version; exercises the lowest-supported boundary
- `gcc g++` via apt — installs `libgomp1` which the ouxinfo wheel needs at runtime; also covers source builds if no wheel matches the runner
- `ouxinfo pytest` installed from PyPI — no local compilation required
- Triggers on all pushes and pull requests (no branch filter) so every branch gets checked

## Verification

After merging, the Actions tab should show a passing "Test" workflow run. Locally, the equivalent is:
```bash
pytest tests/ -v
```
(11 tests, all passing as of this session)
