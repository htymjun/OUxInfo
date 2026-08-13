# Contributing to OUxInfo

Thank you for contributing to OUxInfo.

OUxInfo is a Python package with a C++ backend for fast information-theoretic
estimation. Contributions should prioritize mathematical correctness, numerical
robustness, reproducibility, performance, portability, and a clear Python API.

## Where to start

Please search existing Issues and Discussions first.

- **Bug Report** — incorrect results, crashes, build failures, or unexpected behavior.
- **Feature Request** — a concrete capability to add.
- **Validation / Numerical Issue** — discrepancies with analytical/reference results.
- **Performance Ideas** — optimization and scalability ideas, including speculative ideas.
- **Methodology Discussions** — estimator definitions, mathematics, and research applications.

Rule of thumb:

> Issues describe work that may need to be done. Discussions are for ideas and conversations that are not yet committed work.

## Development workflow

OUxInfo uses a **Fork-based development workflow**. Contributors should not
normally create development branches directly in the main repository.

### 1. Fork the repository

Fork the OUxInfo repository to your own GitHub account.

```text
OUxInfo (upstream)
      |
      +---- Fork ----> your-account/OUxInfo
```

### 2. Clone your fork

```bash
git clone git@github.com:<your-account>/OUxInfo.git
cd OUxInfo
```

### 3. Add the upstream repository

Add the official OUxInfo repository as `upstream`.

```bash
git remote add upstream git@github.com:htymjun/OUxInfo.git
```

### 4. Synchronize with upstream

Before starting development, update your local `main` branch.

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### 5. Create a development branch

Create a separate branch for each change.

```bash
git checkout -b feature/<name>
```

Recommended prefixes are:

```text
feature/<name>
fix/<name>
perf/<name>
validation/<name>
docs/<name>
```

### 6. Develop and test

Make your changes in the development branch and run the relevant tests.

```bash
pytest tests/ -v
```

For changes to information-theoretic estimators, validation against analytical
or reference results is recommended.

### 7. Commit and push

Commit your changes and push the branch to your fork.

```bash
git add .
git commit -m "feat: add new information measure"
git push -u origin <your-branch>
```

### 8. Open a Pull Request

Open a Pull Request from your fork to the official repository.

```text
your-account/OUxInfo:<your-branch>
              |
              | Pull Request
              v
       htymjun/OUxInfo:main
```

Use the Pull Request template and describe the changes, motivation,
implementation, validation, and any numerical or performance impact.

### 9. Update your branch if needed

If `upstream/main` changes while you are developing, update your branch before
continuing.

```bash
git fetch upstream
git checkout main
git merge upstream/main
git checkout <your-branch>
git merge main
```

Resolve any conflicts and rerun the relevant tests.

### 10. Review and merge

Address review comments and push additional commits to the same branch as
needed.

Once the Pull Request is approved, it can be merged into `main` by a repository
maintainer.

## Project architecture

OUxInfo exposes a Python API backed by C++ through pybind11. The package is
built with setuptools and uses OpenMP for parallel execution.

When changing an estimator, consider both the numerical algorithm and its
Python-facing behavior.

## Numerical and statistical correctness

Information-theoretic estimators can be sensitive to sample size, dimension,
k-nearest-neighbor parameters, normalization, boundary effects, finite-sample
bias, and numerical precision.

A change that improves runtime but changes estimator behavior must document the
trade-off explicitly.

## Performance work

OUxInfo uses a C++ backend and OpenMP. Performance changes should be measured.

Where possible report:

- CPU model / architecture
- compiler and version
- OpenMP thread count
- dataset size
- dimensionality
- estimator parameters such as `k`
- runtime before and after
- peak memory when relevant

## Python API compatibility

OUxInfo is distributed through PyPI, so public API compatibility matters.

When changing a public API, update documentation and tests and describe
compatibility implications in the PR.

## Third-party code and licenses

OUxInfo includes third-party components such as Boost and nanoflann. Preserve
their required copyright and license notices.

## AI-assisted development

AI coding assistants may be used. Contributors remain responsible for
mathematical correctness, numerical validity, statistical interpretation,
performance claims, API compatibility, licensing, and reproducibility.

AI-generated code must be reviewed and validated by the contributor.

## Commit messages

Prefer concise messages:

```text
fix: handle degenerate entropy input
feat: add conditional transfer entropy
perf: parallelize nearest-neighbor search
test: validate Gaussian entropy
docs: add transfer entropy example
```

## Review philosophy

Reviews should focus on mathematical correctness, numerical robustness,
statistical behavior, reproducibility, performance, API design, and maintainability.
