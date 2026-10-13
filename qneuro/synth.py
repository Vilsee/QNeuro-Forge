"""Synthetic neurodevelopmental data generator — always labelled SYNTHETIC."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qneuro.models.schemas import DataSource

_FEATURE_NAMES = [
    "verbal_iq",
    "nonverbal_iq",
    "working_memory",
    "processing_speed",
    "social_responsiveness",
    "repetitive_behaviour",
    "sensory_sensitivity",
    "executive_function",
]


def generate(
    n_samples: int = 80,
    n_features: int | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, DataSource]:
    """Generate a balanced binary-classification dataset.

    The distributions are symmetric across classes so that neither quantum
    nor classical models have a structural advantage from data shape alone.
    """
    rng = np.random.default_rng(random_seed)
    if n_features is None:
        n_features = len(_FEATURE_NAMES)
    n_features = min(n_features, len(_FEATURE_NAMES))

    half = n_samples // 2
    labels = np.array([0] * half + [1] * (n_samples - half))

    features = rng.standard_normal((n_samples, n_features))

    # Mild signal: shift two features by 0.5 std for class 1
    # Kept small so neither approach trivially wins
    features[half:, 0] += 0.5
    features[half:, 1] -= 0.5

    cols = _FEATURE_NAMES[:n_features]
    df = pd.DataFrame(features, columns=cols)
    df["diagnosis"] = labels

    return df, DataSource.SYNTHETIC
