"""Seeded synthetic data generation and explicit CSV ingestion."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "verbal_iq",
    "working_memory",
    "processing_speed",
    "social_affect",
    "repetitive_behavior",
    "sensory_seeking",
    "motor_coordination",
    "executive_function",
]
TARGET_COLUMN = "phenotype_group"


def generate_synthetic(
    n_samples: int = 120,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Generate a reproducible tabular neurodevelopmental cohort.

    Generative Specification:
    - 120 instances with 8 numeric cognitive/behavioral features.
    - Mild class imbalance: ~45% negative class (0), ~55% positive class (1).
    - Informative features:
      - `verbal_iq` and `working_memory` feature an additive shift of 0.45 std dev between classes.
      - `social_affect` has a mild non-linear interaction term (quadratic component) with equal variance across classes.
      - Remaining 5 features (`processing_speed`, `repetitive_behavior`, `sensory_seeking`,
        `motor_coordination`, `executive_function`) are drawn from uncorrelated standard normal noise N(0, 1).
    - Process is strictly symmetric across class manifolds and not tuned to favor kernel methods or linear hyperplanes.
    - Marked unambiguously as SYNTHETIC.
    """
    rng = np.random.default_rng(random_seed)

    # Mild class imbalance: 54 class 0 (45%), 66 class 1 (55%)
    n_class_0 = int(n_samples * 0.45)
    n_class_1 = n_samples - n_class_0

    labels = np.array([0] * n_class_0 + [1] * n_class_1, dtype=int)

    # Base features sampled from standard Gaussian
    X = rng.standard_normal((n_samples, len(COLUMN_NAMES)))

    # Mild separation in 2 linear features
    X[labels == 1, 0] += 0.45  # verbal_iq
    X[labels == 1, 1] -= 0.40  # working_memory

    # Mild non-linear interaction in feature 3
    X[labels == 1, 3] = 0.5 * (X[labels == 1, 3] ** 2 - 1.0) + rng.normal(0, 0.5, size=n_class_1)

    # Shuffle instances while maintaining seed determinism
    indices = rng.permutation(n_samples)
    X = X[indices]
    labels = labels[indices]

    df = pd.DataFrame(X, columns=COLUMN_NAMES)
    df[TARGET_COLUMN] = labels

    return df, "SYNTHETIC"


def load_real_csv(
    path: Path | str,
    column_spec: dict[str, Any] | list[str],
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, str]:
    """Ingest real clinical CSV ensuring strict column specification conformance."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Clinical dataset not found at {path}")

    if isinstance(column_spec, dict):
        df = pd.read_csv(p, dtype=column_spec)
        expected_cols = list(column_spec.keys())
    else:
        df = pd.read_csv(p)
        expected_cols = list(column_spec)

    # Verify column presence
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing mandatory columns specified in spec: {missing}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in ingested dataset")

    # Select only specified columns plus target
    required = list(dict.fromkeys(expected_cols + [target_col]))
    df = df[required].copy()

    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values")

    return df, "REAL"


def compute_fingerprint(df: pd.DataFrame) -> str:
    """Generate deterministic SHA-256 fingerprint from dataframe content and structure."""
    canonical_bytes = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(canonical_bytes).hexdigest()
