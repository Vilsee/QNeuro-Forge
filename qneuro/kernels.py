"""Quantum fidelity kernels via PennyLane statevector simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

from qneuro.types import ExperimentConfig

CACHE_DIR = Path(".cache") / "kernels"


def _config_cache_key(config: ExperimentConfig, data_fingerprint: str) -> str:
    """Deterministic hash of configuration and data identity for kernel caching."""
    blob = json.dumps(
        {
            "model_family": config.model_family,
            "feature_map": config.feature_map,
            "n_qubits": config.n_qubits,
            "reps": config.reps,
            "C": config.C,
            "fingerprint": data_fingerprint,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _build_circuit(
    feature_map: str,
    n_qubits: int,
    reps: int,
) -> qml.QNode:
    """Construct a PennyLane circuit returning the full statevector."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray) -> np.ndarray:
        if feature_map == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        elif feature_map == "iqp":
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=reps)
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
        return qml.state()

    return circuit


def compute_kernel_matrix(
    X: np.ndarray,
    config: ExperimentConfig,
    data_fingerprint: str = "",
    use_cache: bool = True,
    cache_label: str = "train",
) -> np.ndarray:
    """Compute symmetric fidelity kernel K[i,j] = |<psi(xi)|psi(xj)>|^2.

    If cache_label='train', looks for / writes a square NxN matrix.
    For rectangular cross-kernel matrices use compute_cross_kernel instead.
    """
    cache_key = _config_cache_key(config, data_fingerprint) + f"_{cache_label}"
    cache_path = CACHE_DIR / f"{cache_key}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    circuit = _build_circuit(config.feature_map or "angle", config.n_qubits, config.reps)

    n = X.shape[0]
    # Truncate or pad features to match qubit count
    X_q = _prepare_features(X, config.n_qubits)

    states = np.array([circuit(X_q[i]) for i in range(n)])

    # Fidelity kernel from inner products of statevectors
    K = np.abs(states @ states.conj().T) ** 2
    # Ensure exact symmetry from floating point
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 1.0)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, K)

    return K


def compute_cross_kernel(
    X_test: np.ndarray,
    X_train: np.ndarray,
    config: ExperimentConfig,
    data_fingerprint: str = "",
    use_cache: bool = True,
) -> np.ndarray:
    """Compute rectangular kernel K[i,j] = |<psi(x_test_i)|psi(x_train_j)>|^2."""
    cache_key = _config_cache_key(config, data_fingerprint) + "_cross"
    cache_path = CACHE_DIR / f"{cache_key}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    circuit = _build_circuit(config.feature_map or "angle", config.n_qubits, config.reps)

    X_test_q = _prepare_features(X_test, config.n_qubits)
    X_train_q = _prepare_features(X_train, config.n_qubits)

    states_test = np.array([circuit(X_test_q[i]) for i in range(X_test_q.shape[0])])
    states_train = np.array([circuit(X_train_q[i]) for i in range(X_train_q.shape[0])])

    K = np.abs(states_test @ states_train.conj().T) ** 2

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, K)

    return K


def _prepare_features(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Truncate or zero-pad feature vectors to match qubit register width."""
    n_features = X.shape[1]
    if n_features >= n_qubits:
        return X[:, :n_qubits]
    # Pad with zeros when fewer features than qubits
    pad_width = n_qubits - n_features
    return np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def fit_fold_scaler(
    X_train: np.ndarray,
) -> MinMaxScaler:
    """Fit a scaler on the training fold only, mapping to [0, pi] for circuit input."""
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    scaler.fit(X_train)
    return scaler


def quantum_configs() -> list[ExperimentConfig]:
    """Enumerate default quantum experiment configurations."""
    configs = []
    for fm in ("angle", "iqp"):
        for nq in (4, 6):
            reps = 2 if fm == "iqp" else 1
            configs.append(
                ExperimentConfig(
                    model_family="quantum_svc",
                    feature_map=fm,
                    n_qubits=nq,
                    reps=reps,
                    C=1.0,
                )
            )
    return configs
