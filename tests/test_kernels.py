"""Test quantum kernel matrix mathematical properties."""
from __future__ import annotations

import numpy as np
import pytest

from qneuro.data import generate_synthetic, compute_fingerprint, COLUMN_NAMES
from qneuro.kernels import (
    compute_kernel_matrix,
    compute_cross_kernel,
    fit_fold_scaler,
    _prepare_features,
)
from qneuro.types import ExperimentConfig

# Use a small subset for fast tests
_SEED = 42
_N_QUBITS = 4
_TOLERANCE = 1e-8


def _make_config(feature_map: str = "angle") -> ExperimentConfig:
    return ExperimentConfig(
        model_family="quantum_svc",
        feature_map=feature_map,
        n_qubits=_N_QUBITS,
        reps=2,
        C=1.0,
    )


@pytest.fixture
def small_dataset() -> tuple[np.ndarray, str]:
    """Generate 20 rows for kernel property tests."""
    df, source = generate_synthetic(n_samples=20, random_seed=_SEED)
    X = df[COLUMN_NAMES].to_numpy()
    scaler = fit_fold_scaler(X)
    X_scaled = scaler.transform(X)
    fingerprint = compute_fingerprint(df)
    return X_scaled, fingerprint


@pytest.mark.parametrize("feature_map", ["angle", "iqp"])
def test_kernel_symmetric(small_dataset: tuple[np.ndarray, str], feature_map: str) -> None:
    """Kernel matrix K must equal its transpose within floating-point tolerance."""
    X, fp = small_dataset
    config = _make_config(feature_map)
    K = compute_kernel_matrix(X, config, data_fingerprint=fp, use_cache=False)
    np.testing.assert_allclose(K, K.T, atol=_TOLERANCE)


@pytest.mark.parametrize("feature_map", ["angle", "iqp"])
def test_kernel_diagonal_ones(small_dataset: tuple[np.ndarray, str], feature_map: str) -> None:
    """Self-fidelity of any state is 1: K[i,i] = |<psi(xi)|psi(xi)>|^2 = 1."""
    X, fp = small_dataset
    config = _make_config(feature_map)
    K = compute_kernel_matrix(X, config, data_fingerprint=fp, use_cache=False)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=_TOLERANCE)


@pytest.mark.parametrize("feature_map", ["angle", "iqp"])
def test_kernel_positive_semidefinite(small_dataset: tuple[np.ndarray, str], feature_map: str) -> None:
    """Fidelity kernel is a Gram matrix, hence all eigenvalues >= 0."""
    X, fp = small_dataset
    config = _make_config(feature_map)
    K = compute_kernel_matrix(X, config, data_fingerprint=fp, use_cache=False)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues >= -_TOLERANCE), (
        f"Negative eigenvalue below tolerance: {eigenvalues.min():.2e}"
    )


def test_kernel_values_bounded(small_dataset: tuple[np.ndarray, str]) -> None:
    """All fidelity values lie in [0, 1]."""
    X, fp = small_dataset
    config = _make_config("angle")
    K = compute_kernel_matrix(X, config, data_fingerprint=fp, use_cache=False)
    assert K.min() >= -_TOLERANCE
    assert K.max() <= 1.0 + _TOLERANCE


def test_cross_kernel_shape(small_dataset: tuple[np.ndarray, str]) -> None:
    """Cross-kernel between test and train has shape (n_test, n_train)."""
    X, fp = small_dataset
    config = _make_config("angle")
    X_train = X[:15]
    X_test = X[15:]
    K_cross = compute_cross_kernel(X_test, X_train, config, data_fingerprint=fp, use_cache=False)
    assert K_cross.shape == (5, 15)


def test_prepare_features_truncation() -> None:
    """Features wider than qubit count are truncated to n_qubits columns."""
    X = np.random.default_rng(0).random((5, 8))
    X_q = _prepare_features(X, 4)
    assert X_q.shape == (5, 4)
    np.testing.assert_array_equal(X_q, X[:, :4])


def test_prepare_features_padding() -> None:
    """Features narrower than qubit count are zero-padded."""
    X = np.random.default_rng(0).random((5, 3))
    X_q = _prepare_features(X, 6)
    assert X_q.shape == (5, 6)
    np.testing.assert_array_equal(X_q[:, 3:], 0.0)
