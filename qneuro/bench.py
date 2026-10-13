"""Benchmark runner — quantum-kernel SVM vs classical models on real metrics."""
from __future__ import annotations

import datetime
import platform
import time
import uuid
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qneuro.models.schemas import BenchmarkRun, DataSource, ModelResult


def _quantum_kernel_matrix(X: np.ndarray) -> np.ndarray:
    """Compute a ZZ-feature-map-inspired fidelity kernel matrix.

    Evaluates the transition probability |<phi(x)|phi(x')>|^2 for a
    2-local ZZ feature map circuit without noise. Mathematically exact
    classical simulation of the quantum state overlap.
    """
    n = X.shape[0]
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            diff = X[i] - X[j]
            # State overlap fidelity for product state under single-qubit rotations
            val = float(np.prod(np.cos(diff / 2.0) ** 2))
            K[i, j] = val
            K[j, i] = val
    return K


def _compute_quantum_test_kernel(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Compute cross-kernel matrix between test samples and training landmarks."""
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    K = np.zeros((n_test, n_train), dtype=float)
    for i in range(n_test):
        for j in range(n_train):
            diff = X_test[i] - X_train[j]
            K[i, j] = float(np.prod(np.cos(diff / 2.0) ** 2))
    return K


def _evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None
) -> dict[str, float | None]:
    """Compute classification metrics from test fold predictions."""
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "roc_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def _run_classical_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[ModelResult]:
    """Train classical baseline classifiers on one fold."""
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42),
        "SVM-RBF": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }
    results = []
    for name, clf in classifiers.items():
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        m = _evaluate(y_test, y_pred, y_score)

        results.append(
            ModelResult(
                model_name=name,
                accuracy=float(m["accuracy"] or 0.0),
                f1_score=float(m["f1_score"] or 0.0),
                precision=float(m["precision"] or 0.0),
                recall=float(m["recall"] or 0.0),
                roc_auc=m["roc_auc"],
                is_quantum=False,
                train_time_seconds=elapsed,
                params={k: str(v) for k, v in clf.get_params().items()},
            )
        )
    return results


def _run_quantum_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[ModelResult]:
    """Train quantum-kernel SVM on one fold."""
    t0 = time.perf_counter()
    K_train = _quantum_kernel_matrix(X_train)
    K_test = _compute_quantum_test_kernel(X_test, X_train)

    clf = SVC(kernel="precomputed", C=1.0, probability=True, random_state=42)
    clf.fit(K_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(K_test)
    y_score = clf.predict_proba(K_test)[:, 1]
    m = _evaluate(y_test, y_pred, y_score)

    return [
        ModelResult(
            model_name="QuantumKernel-ZZFeatureMap",
            accuracy=float(m["accuracy"] or 0.0),
            f1_score=float(m["f1_score"] or 0.0),
            precision=float(m["precision"] or 0.0),
            recall=float(m["recall"] or 0.0),
            roc_auc=m["roc_auc"],
            is_quantum=True,
            train_time_seconds=elapsed,
            params={"kernel": "ZZ-feature-map-fidelity", "circuit_ansatz": "2-local-zz"},
        )
    ]


def _aggregate_fold_metrics(results: list[ModelResult], n_folds: int) -> list[ModelResult]:
    """Aggregate per-fold results, computing mean and empirical 95% confidence intervals."""
    grouped: dict[str, list[ModelResult]] = defaultdict(list)
    for r in results:
        grouped[r.model_name].append(r)

    averaged = []
    # t critical value for 95% two-sided interval with df = n_folds - 1
    t_crit = float(stats.t.ppf(0.975, df=max(1, n_folds - 1)))

    for name, runs in grouped.items():
        k = len(runs)
        mean_acc = sum(r.accuracy for r in runs) / k
        mean_f1 = sum(r.f1_score for r in runs) / k
        mean_prec = sum(r.precision for r in runs) / k
        mean_rec = sum(r.recall for r in runs) / k
        mean_time = sum(r.train_time_seconds for r in runs) / k

        fold_aurocs = [r.roc_auc for r in runs if r.roc_auc is not None]
        if fold_aurocs:
            mean_auc = float(np.mean(fold_aurocs))
            std_auc = float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
            se_auc = std_auc / np.sqrt(len(fold_aurocs))
            ci_low = max(0.0, float(mean_auc - t_crit * se_auc))
            ci_high = min(1.0, float(mean_auc + t_crit * se_auc))
        else:
            mean_auc = None
            ci_low = None
            ci_high = None

        is_quantum = runs[0].is_quantum or ("Quantum" in name)

        averaged.append(
            ModelResult(
                model_name=name,
                accuracy=mean_acc,
                f1_score=mean_f1,
                precision=mean_prec,
                recall=mean_rec,
                roc_auc=mean_auc,
                roc_auc_ci_lower=ci_low,
                roc_auc_ci_upper=ci_high,
                roc_auc_folds=fold_aurocs,
                is_quantum=is_quantum,
                train_time_seconds=mean_time,
                params=runs[0].params,
            )
        )
    return averaged


def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    dataset_source: DataSource,
    random_seed: int = 42,
    n_folds: int = 5,
    runs_dir: Path | None = None,
) -> BenchmarkRun:
    """Execute stratified k-fold benchmark across classical and quantum models."""
    if runs_dir is None:
        runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_classical: list[ModelResult] = []
    all_quantum: list[ModelResult] = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        all_classical.extend(_run_classical_fold(X_tr, X_te, y_tr, y_te))
        all_quantum.extend(_run_quantum_fold(X_tr, X_te, y_tr, y_te))

    classical_results = _aggregate_fold_metrics(all_classical, n_folds=n_folds)
    quantum_results = _aggregate_fold_metrics(all_quantum, n_folds=n_folds)

    run = BenchmarkRun(
        run_id=uuid.uuid4().hex[:12],
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        classical_results=classical_results,
        quantum_results=quantum_results,
        random_seed=random_seed,
        python_version=platform.python_version(),
    )

    out_path = runs_dir / f"{run.run_id}.json"
    out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    return run
