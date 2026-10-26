"""Exact NumPy statevector kernels used in the Morphos hackathon study.

Two kernels appear in the study and they are not the same thing:

* ``zz_states`` builds the standard ZZ feature map state (Hadamards, single-qubit
  phases 2x_i, and a pair phase 2(pi - x_i)(pi - x_j) on every qubit pair), which
  entangles the qubits.
* ``product_kernel`` is the closed form of an angle-embedding (RX) fidelity kernel.
  The state is a product state, so the overlap factorises per qubit.
"""
from __future__ import annotations

import itertools

import numpy as np


def _hadamard_n(n: int) -> np.ndarray:
    h1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hn = h1
    for _ in range(n - 1):
        hn = np.kron(hn, h1)
    return hn


def zz_states(X: np.ndarray, reps: int = 1) -> np.ndarray:
    """Return one ZZ-feature-map statevector per row of ``X`` (4 qubits for 4 features)."""
    n = X.shape[1]
    hn = _hadamard_n(n)
    bits = np.array(list(itertools.product([0, 1], repeat=n)))
    pairs = list(itertools.combinations(range(n), 2))
    out = []
    for x in X:
        ph = (bits * 2 * x).sum(1) + sum(
            (bits[:, i] ^ bits[:, j]) * 2 * (np.pi - x[i]) * (np.pi - x[j]) for i, j in pairs
        )
        s = np.zeros(2**n, complex)
        s[0] = 1
        for _ in range(reps):
            s = np.exp(1j * ph) * (hn @ s)
        out.append(s)
    return np.array(out)


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """K(a, b) = |<phi(a)|phi(b)>|^2 from precomputed statevectors."""
    return np.abs(states_a.conj() @ states_b.T) ** 2


def zz_kernel(A: np.ndarray, B: np.ndarray, reps: int = 1) -> np.ndarray:
    return fidelity_kernel(zz_states(A, reps), zz_states(B, reps))


def product_kernel(A: np.ndarray, B: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Angle-embedding fidelity kernel, closed form: prod_i cos^2((a_i - b_i) * scale / 2)."""
    d = (A[:, None, :] - B[None, :, :]) * scale
    return np.prod(np.cos(d / 2) ** 2, axis=2)
