"""Classical ML baselines and unified benchmark orchestrator using types.py models."""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
from xgboost import XGBClassifier

from qneuro.data import compute_fingerprint, COLUMN_NAMES, TARGET_COLUMN
from qneuro.kernels import (
    compute_cross_kernel,
    compute_kernel_matrix,
    fit_fold_scaler,
    quantum_configs,
)
from qneuro.types import ExperimentConfig, FoldScores


def _classical_configs() -> list[tuple[ExperimentConfig, Any]]:
    """Enumerate classical model configurations with inner-CV parameter grids.

    Returns (config, param_grid) pairs where param_grid drives GridSearchCV.
    The hyperparameter budget is comparable to the quantum side's single-C SVC.
    """
    return [
        (
            ExperimentConfig(model_family="rbf_svc", C=1.0),
            {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
            },
        ),
        (
            ExperimentConfig(model_family="logistic_regression", C=1.0),
            {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            },
        ),
        (
            ExperimentConfig(model_family="xgboost", C=1.0),
            {
                "max_depth": [3, 5],
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.3],
            },
        ),
    ]


def _make_classical_estimator(config: ExperimentConfig) -> Any:
    """Instantiate the sklearn/xgboost estimator for a given config."""
    if config.model_family == "rbf_svc":
        return SVC(kernel="rbf", probability=True, random_state=42)
    if config.model_family == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if config.model_family == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    raise ValueError(f"Unknown classical family: {config.model_family}")


def run_full_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 3,
    data_fingerprint: str = "",
) -> tuple[list[FoldScores], dict[str, float]]:
    """Run all quantum and classical models on identical RepeatedStratifiedKFold splits.

    Returns:
        fold_scores: List of FoldScores with identical fold indexing across all models.
        wall_clocks: Map from config_id to total wall-clock seconds.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed
    )
    folds = list(rskf.split(X, y))
    n_folds = len(folds)

    all_scores: list[FoldScores] = []
    wall_clocks: dict[str, float] = {}

    for qcfg in quantum_configs():
        auroc_per_fold: list[float] = []
        f1_per_fold: list[float] = []
        acc_per_fold: list[float] = []
        t0 = time.perf_counter()

        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Scale inside the fold to prevent leakage
            scaler = fit_fold_scaler(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            K_train = compute_kernel_matrix(
                X_tr_s, qcfg, data_fingerprint=data_fingerprint, use_cache=False
            )
            K_test = compute_cross_kernel(
                X_te_s, X_tr_s, qcfg, data_fingerprint=data_fingerprint, use_cache=False
            )

            clf = SVC(kernel="precomputed", C=qcfg.C, probability=True, random_state=42)
            clf.fit(K_train, y_tr)

            y_score = clf.predict_proba(K_test)[:, 1]
            y_pred = clf.predict(K_test)

            auroc_per_fold.append(float(roc_auc_score(y_te, y_score)))
            f1_per_fold.append(float(f1_score(y_te, y_pred, zero_division=0.0)))
            acc_per_fold.append(float(accuracy_score(y_te, y_pred)))

        elapsed = time.perf_counter() - t0
        cid = qcfg.config_id
        wall_clocks[cid] = elapsed

        for metric_name, scores in [
            ("auroc", auroc_per_fold),
            ("f1", f1_per_fold),
            ("accuracy", acc_per_fold),
        ]:
            arr = np.array(scores)
            ci_l, ci_u = _ci_95(arr)
            all_scores.append(
                FoldScores(
                    config_id=cid,
                    metric=metric_name,
                    scores=scores,
                    mean_score=float(arr.mean()),
                    std_score=float(arr.std(ddof=1)),
                    ci_lower=ci_l,
                    ci_upper=ci_u,
                )
            )

    inner_cv = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=1, random_state=random_seed
    )

    for ccfg, param_grid in _classical_configs():
        auroc_per_fold = []
        f1_per_fold = []
        acc_per_fold = []
        t0 = time.perf_counter()

        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            base_est = _make_classical_estimator(ccfg)
            gs = GridSearchCV(
                base_est,
                param_grid,
                cv=inner_cv,
                scoring="roc_auc",
                refit=True,
                n_jobs=1,
            )
            gs.fit(X_tr_s, y_tr)
            best = gs.best_estimator_

            y_pred = best.predict(X_te_s)
            if hasattr(best, "predict_proba"):
                y_score = best.predict_proba(X_te_s)[:, 1]
            else:
                y_score = best.decision_function(X_te_s)

            auroc_per_fold.append(float(roc_auc_score(y_te, y_score)))
            f1_per_fold.append(float(f1_score(y_te, y_pred, zero_division=0.0)))
            acc_per_fold.append(float(accuracy_score(y_te, y_pred)))

        elapsed = time.perf_counter() - t0
        cid = ccfg.config_id
        wall_clocks[cid] = elapsed

        for metric_name, scores in [
            ("auroc", auroc_per_fold),
            ("f1", f1_per_fold),
            ("accuracy", acc_per_fold),
        ]:
            arr = np.array(scores)
            ci_l, ci_u = _ci_95(arr)
            all_scores.append(
                FoldScores(
                    config_id=cid,
                    metric=metric_name,
                    scores=scores,
                    mean_score=float(arr.mean()),
                    std_score=float(arr.std(ddof=1)),
                    ci_lower=ci_l,
                    ci_upper=ci_u,
                )
            )

    return all_scores, wall_clocks


def _ci_95(arr: np.ndarray) -> tuple[float, float]:
    """Compute 95% t-distribution confidence interval for a sample array."""
    n = len(arr)
    if n < 2:
        return float(arr[0]), float(arr[0])
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return max(0.0, mean - t_crit * se), min(1.0, mean + t_crit * se)
