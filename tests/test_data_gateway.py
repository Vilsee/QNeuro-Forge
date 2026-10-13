"""Tests for synthetic data determinism and strict gateway privacy isolation."""
from __future__ import annotations

import pandas as pd
import pytest

from qneuro.data import COLUMN_NAMES, TARGET_COLUMN, compute_fingerprint, generate_synthetic, load_real_csv
from qneuro.gateway import DatasetGateway, GatewayReport


def test_synthetic_determinism() -> None:
    """Verify identical random seeds produce byte-identical datasets."""
    df1, src1 = generate_synthetic(n_samples=120, random_seed=42)
    df2, src2 = generate_synthetic(n_samples=120, random_seed=42)

    assert src1 == "SYNTHETIC"
    assert src2 == "SYNTHETIC"
    pd.testing.assert_frame_equal(df1, df2)
    assert compute_fingerprint(df1) == compute_fingerprint(df2)

    df_other, _ = generate_synthetic(n_samples=120, random_seed=99)
    assert compute_fingerprint(df1) != compute_fingerprint(df_other)


def test_synthetic_specification() -> None:
    """Validate structure, feature count, and class distribution constraints."""
    df, src = generate_synthetic(n_samples=120, random_seed=42)

    assert src == "SYNTHETIC"
    assert len(df) == 120
    assert list(df.columns) == COLUMN_NAMES + [TARGET_COLUMN]

    # Mild imbalance check: between 40% and 60%
    class_1_ratio = (df[TARGET_COLUMN] == 1).mean()
    assert 0.45 <= class_1_ratio <= 0.60


def test_gateway_leaks_no_raw_rows() -> None:
    """Confirm the gateway exposes only aggregates and shields individual rows."""
    df, src = generate_synthetic(n_samples=120, random_seed=42)
    gw = DatasetGateway(df, source=src, target_col=TARGET_COLUMN)

    # Public interface must only expose permitted summary methods
    public_attrs = [attr for attr in dir(gw) if not attr.startswith("_")]
    permitted = {"describe", "get_class_balance", "get_feature_summaries", "get_row_count", "get_schema"}
    assert set(public_attrs) == permitted

    # Disallow direct row access operators
    assert not hasattr(gw, "iloc")
    assert not hasattr(gw, "loc")
    assert not hasattr(gw, "values")
    assert not hasattr(gw, "__getitem__")
    assert not hasattr(gw, "__iter__")

    # Outputs must be strictly bounded summary models
    report = gw.describe()
    assert isinstance(report, GatewayReport)
    assert report.row_count == 120
    assert report.dataset_source == "SYNTHETIC"

    # Verify no raw individual row values are contained in summaries
    for feat_sum in report.feature_summaries:
        assert isinstance(feat_sum.mean, float)
        assert isinstance(feat_sum.std, float)
        assert not hasattr(feat_sum, "values")
        assert not hasattr(feat_sum, "data")


def test_real_csv_validation(tmp_path) -> None:
    """Confirm CSV loader strictly checks column specifications."""
    df, _ = generate_synthetic(n_samples=50, random_seed=12)
    csv_file = tmp_path / "valid.csv"
    df.to_csv(csv_file, index=False)

    loaded, src = load_real_csv(csv_file, column_spec=COLUMN_NAMES, target_col=TARGET_COLUMN)
    assert src == "REAL"
    assert len(loaded) == 50

    # Incomplete specification should trigger ValueError
    with pytest.raises(ValueError, match="missing mandatory columns"):
        load_real_csv(csv_file, column_spec=COLUMN_NAMES + ["missing_marker"], target_col=TARGET_COLUMN)
