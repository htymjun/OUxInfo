# Plan: GitHub Actions CI for pytest

## Goal

Run `pytest tests/` automatically on every push and pull request so regressions are caught without manual effort.

## Approach

Install `ouxinfo` from PyPI (binary wheel) plus `pytest`, then run the test suite. GCC and G++ are installed via apt to satisfy the OpenMP runtime dependency (`libgomp1`) that the wheel links against.

## File

`.github/workflows/test.yml`

- Runner: `ubuntu-latest`
- Python: `3.12` (minimum supported version)
- Installs: `gcc g++` (apt), `ouxinfo pytest` (pip)
- Command: `pytest tests/ -v`
- Trigger: all pushes and pull requests
