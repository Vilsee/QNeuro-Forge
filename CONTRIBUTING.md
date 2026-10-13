# Contributing to QNeuro Forge

We welcome contributions to QNeuro Forge. Please review the core principles in `AGENTS.md` and `DESIGN.md` before submitting code.

## Core Directives

1. **Non-Negotiable Empirical Truth**: Never mock, hardcode, or edit benchmark figures, metrics, or p-values. All values must be produced by real verifiable runs written to `runs/`.
2. **Gateway Privacy**: Code that touches tabular datasets must respect the `DatasetGateway` boundary. The agent context must never inspect raw patient rows.
3. **Guard Monopoly**: Only `qneuro/guard.py` is authorized to formulate or decide empirical superiority arbitration claims.

---

## Good First Issues

### Issue 1: Add Disk Cache Eviction Policy for Kernel Matrices
- **Context**: `qneuro/kernels.py` caches computed Gram matrices in `.cache/kernels/` keyed by configuration hash and dataset fingerprint. Over extensive agent exploration sessions, cached `.npy` files can accumulate indefinitely.
- **Task**: Implement a Least-Recently-Used (LRU) or maximum-disk-quota cache cleaner in `qneuro/kernels.py` that limits cache size to a configurable maximum (e.g. 500 MB) without deleting active fold matrices.
- **Difficulty**: Beginner
- **Skills**: Python, `pathlib`, file system management.

### Issue 2: Implement Permutation Feature Importance for Classical Baselines
- **Context**: `qneuro/baselines.py` evaluates models across stratified folds but does not report feature importance scores to the gateway.
- **Task**: Add a helper in `qneuro/baselines.py` that computes out-of-fold permutation importance on the validation fold for `rbf_svc`, `logistic_regression`, and `xgboost`, returning mean importance per feature without exposing raw feature distributions.
- **Difficulty**: Intermediate
- **Skills**: `scikit-learn`, statistical aggregation.

### Issue 3: Support Missing-Value Imputation Strategies in CSV Loader
- **Context**: `qneuro/data.py` provides `load_real_csv()` with strict column schema validation, but raises an error if numeric features contain null values.
- **Task**: Add configurable within-fold median and iterative imputation options to `load_real_csv()` and `fit_fold_scaler()`, ensuring that imputation statistics are computed strictly on training partitions to prevent data leakage.
- **Difficulty**: Intermediate
- **Skills**: `pandas`, preprocessing, cross-validation isolation.

---

## Submitting Pull Requests

1. Fork the repository and create your feature branch from `main`.
2. Ensure full type hints are present (mypy clean) and Pydantic v2 models are maintained at interface boundaries.
3. Verify that all tests pass: `pytest -v`.
4. Submit your pull request with a concise description explaining why the change was made.
